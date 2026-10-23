"""核心行为回归测试 / Behavioral regression tests for the UW-DNeRF migration."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uw_losses import uncertainty_depth_loss
from uw_data import load_scalar_maps, normalize_uncertainty, temporal_split
from uw_metrics import depth_metrics
from run_endonerf_helpers import DNeRFPlus, get_embedder, NeRF


def model(resolution=6, keys=(0., .5), use_key_frame=True):
    embed, channels = get_embedder(resolution, 3)
    time_embed, time_channels = get_embedder(resolution, 1)
    network = DNeRFPlus(D=4, W=16, skips=(), D_t=2, D_d=2,
                        input_ch=channels, input_ch_views=0, input_ch_time=time_channels,
                        embed_fn=embed, embedtime_fn=time_embed, key_frame_times=keys,
                        use_key_frame=use_key_frame)
    return network, embed, time_embed


class LossTests(unittest.TestCase):
    def test_inside_and_boundary_are_differentiable_zero(self):
        for values in ([1., 2.], [1.5, 2.5]):
            pred = torch.tensor(values, requires_grad=True)
            loss = uncertainty_depth_loss(pred, torch.tensor([1., 2.]), torch.tensor([.5, .5]))
            self.assertEqual(loss.item(), 0.)
            loss.backward()
            torch.testing.assert_close(pred.grad, torch.zeros(2))

    def test_active_gradient_and_reductions(self):
        pred = torch.tensor([1., 2.], requires_grad=True)
        prior, std = torch.tensor([1., 1.]), torch.tensor([.1, .1])
        loss = uncertainty_depth_loss(pred, prior, std)
        self.assertAlmostEqual(loss.item(), .0198, places=6)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([0., .02]))
        self.assertAlmostEqual(uncertainty_depth_loss(pred, prior, std, kind='l1', reduction='mean').item(), .5)

    def test_mask_and_shape_validation(self):
        pred = torch.tensor([4., 2.], requires_grad=True)
        loss = uncertainty_depth_loss(pred, torch.tensor([1., 2.]), torch.zeros(2), torch.tensor([False, True]))
        self.assertEqual(loss.item(), 0.)
        with self.assertRaises(ValueError):
            uncertainty_depth_loss(pred, pred, torch.zeros(2, 1))


class ModelTests(unittest.TestCase):
    def test_dynamic_encoding_and_all_network_gradients(self):
        for resolution in (3, 6, 10):
            network, embed, et = model(resolution)
            points = embed(torch.rand(4, 3))
            times = et(torch.full((4, 1), .25))
            out, delta = network(points, [times, times])
            self.assertEqual(tuple(out.shape), (4, 4))
            (out.square().sum() + delta.square().sum()).backward()
            for prefix in ('_occ', '_timenet', '_deformnet'):
                grads = [p.grad for name, p in network.named_parameters() if name.startswith(prefix)]
                active_grads = [g for g in grads if g is not None]
                self.assertTrue(active_grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in active_grads))
                self.assertTrue(any(g.abs().sum() > 0 for g in active_grads))

    def test_interpolation_tail_and_canonical(self):
        network, embed, et = model()
        # 常量位移探针核验组合/插值，而非只检查张量尺寸。
        # Analytic displacement probes validate composition and interpolation numerically.
        def query(points, source, target, layers, output):
            value = 2. if layers is network._deformnet else 4.
            return points[:, :3].new_full((points.shape[0], 3), value)
        network.query_time = query
        for time, expected in [(0., 0.), (.25, 4.), (.5, 4.), (.75, 6.)]:
            times = et(torch.full((3, 1), time))
            _, delta = network(embed(torch.rand(3, 3)), [times, times])
            torch.testing.assert_close(delta, torch.full((3, 3), expected))

    def test_checkpoint_restores_keyframe_schedule_and_output(self):
        network, embed, et = model(keys=(0., .3, .6, .9))
        restored, _, _ = model(keys=(0.,))
        restored.load_state_dict(network.state_dict())
        times, points = et(torch.full((2, 1), .7)), embed(torch.rand(2, 3))
        for a, b in zip(network(points, [times, times]), restored(points, [times, times])):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(restored.key_frame_times, network.key_frame_times)

    def test_single_key_and_disabled_window(self):
        for keys, enabled in [((0.,), True), ((0., .5), False)]:
            network, embed, et = model(keys=keys, use_key_frame=enabled)
            times = et(torch.full((2, 1), .75))
            out, delta = network(embed(torch.rand(2, 3)), [times, times])
            self.assertTrue(torch.isfinite(out).all() and torch.isfinite(delta).all())

    def test_legacy_model_still_constructs(self):
        embed, channels = get_embedder(3, 3)
        et, tc = get_embedder(3, 1)
        network = NeRF.get_by_name('direct_temporal', D=4, W=16, skips=(), input_ch=channels,
                                  input_ch_views=0, input_ch_time=tc, embed_fn=embed, embedtime_fn=et)
        times = et(torch.full((2, 1), .5))
        out, _ = network(embed(torch.rand(2, 3)), [times, times])
        self.assertEqual(tuple(out.shape), (2, 4))


class DataTests(unittest.TestCase):
    def test_depth_metrics_and_explicit_scale_alignment(self):
        result = depth_metrics(np.array([2., 4.]), np.array([1., 2.]))
        self.assertAlmostEqual(result['abs_rel'], 1.)
        self.assertAlmostEqual(result['sq_rel'], 1.5)
        self.assertAlmostEqual(result['rmse'], np.sqrt(2.5))
        self.assertEqual(depth_metrics(np.array([2., 4.]), np.array([1., 2.]), align_median=True)['rmse'], 0.)
    def test_constant_and_raw_uncertainty(self):
        constant = np.ones((2, 3, 4), dtype=np.float32)
        np.testing.assert_allclose(normalize_uncertainty(constant), 1e-5)
        np.testing.assert_array_equal(normalize_uncertainty(constant, mode='raw'), constant)
        normalized = normalize_uncertainty(np.arange(6, dtype=np.float32))
        self.assertAlmostEqual(float(normalized.min()), 1e-5)
        self.assertAlmostEqual(float(normalized.max()), 1e-4)

    def test_split_has_no_leakage_and_matches_profiles(self):
        train, test = temporal_split(63, 8, 1)
        self.assertEqual(test.tolist(), [1, 9, 17, 25, 33, 41, 49, 57])
        self.assertFalse(set(train) & set(test))
        _, paper_test = temporal_split(63, 8, 7)
        self.assertEqual(paper_test.tolist(), [7, 15, 23, 31, 39, 47, 55])

    def test_offline_maps_missing_count_and_values(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_scalar_maps(folder, 'std', (2, 3, 4))
            directory = Path(folder) / 'std'
            directory.mkdir()
            np.save(directory / '0.npy', np.ones((3, 4), dtype=np.float32))
            with self.assertRaises(ValueError):
                load_scalar_maps(folder, 'std', (2, 3, 4))
            np.save(directory / '1.npy', np.full((3, 4), 2., dtype=np.float32))
            maps = load_scalar_maps(folder, 'std', (2, 3, 4))
            self.assertEqual(maps.shape, (2, 3, 4))
            np.testing.assert_array_equal(maps[1], 2.)


class TrainingIntegrationTests(unittest.TestCase):
    def test_bilingual_configuration_profiles(self):
        import run_endonerf as runner
        directory = Path(__file__).resolve().parents[1] / 'configs'
        for name in ('example.txt', 'uw_dnerf.txt', 'uw_dnerf_paper_settings.txt'):
            args = runner.config_parser().parse_args(['--config', str(directory / name)])
            if name == 'example.txt':
                self.assertEqual(args.nerf_type, 'direct_temporal')
                self.assertFalse(args.use_std_as_depth_tolerance)
            else:
                self.assertEqual(args.nerf_type, 'dnerfplus')
                self.assertTrue(args.use_key_frame and args.use_std_as_depth_tolerance)
        self.assertEqual(args.uncertainty_loss, 'l1')
        self.assertEqual(args.test_frame_offset, 7)

    def test_fine_model_parameters_are_optimized(self):
        import run_endonerf as runner
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'fine').mkdir()
            args = runner.config_parser().parse_args([
                '--basedir', folder, '--expname', 'fine', '--nerf_type', 'dnerfplus',
                '--use_key_frame', '--use_two_models_for_fine', '--netwidth', '16',
                '--netwidth_fine', '16', '--netdepth', '4', '--netdepth_fine', '4',
                '--D_t', '2', '--D_d', '2', '--multires', '3', '--no_reload'])
            train, _, _, _, optimizer, _ = runner.create_nerf(args)
            expected = {id(p) for name in ('network_fn', 'network_fine') for p in train[name].parameters()}
            actual = [id(p) for group in optimizer.param_groups for p in group['params']]
            self.assertEqual(set(actual), expected)
            self.assertEqual(len(actual), len(expected))

    def test_cpu_training_render_reload_and_legacy_baseline(self):
        import run_endonerf as runner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('images', 'pred_depth', 'std', 'masks'):
                (root / name).mkdir()
            pose = np.concatenate((np.eye(3), np.zeros((3, 1)), np.array([[8.], [8.], [8.]])), axis=1)
            poses = np.tile(np.r_[pose.reshape(-1), 1., 3.], (9, 1))
            np.save(root / 'poses_bounds.npy', poses)
            for frame in range(9):
                imageio.imwrite(root / 'images' / ('%03d.png' % frame), np.full((8, 8, 3), 100 + frame, dtype=np.uint8))
                mask = np.zeros((8, 8), dtype=np.uint8)
                mask[0] = 255
                imageio.imwrite(root / 'masks' / ('%03d.png' % frame), mask)
                np.save(root / 'pred_depth' / ('%03d.npy' % frame), np.full((8, 8), 2., dtype=np.float32))
                np.save(root / 'std' / ('%03d.npy' % frame), np.arange(64, dtype=np.float32).reshape(8, 8))
            argv = ['run_endonerf.py', '--datadir', folder, '--basedir', str(root / 'logs'),
                    '--expname', 'smoke', '--nerf_type', 'dnerfplus', '--use_key_frame',
                    '--use_viewdirs', '--use_fgmask', '--mask_loss',
                    '--use_std_as_depth_tolerance', '--use_depth', '--depth_dir', 'pred_depth',
                    '--no_depth_refine', '--no_pose_scale', '--no_ndc', '--no_batching',
                    '--davinci_endoscopic', '--llff_renderpath', 'fixidentity', '--factor', '1',
                    '--test_train_split', '8', '--key_frame_interval', '3', '--N_iter', '1',
                    '--N_rand', '8', '--N_samples', '8', '--N_importance', '4',
                    '--netdepth', '4', '--netwidth', '16', '--D_t', '2', '--D_d', '2',
                    '--multires', '3', '--i_weights', '1', '--i_print', '1']
            with patch.object(sys, 'argv', argv):
                runner.train()
            checkpoint = root / 'logs' / 'smoke' / '000001.tar'
            self.assertTrue(checkpoint.exists())
            args = runner.config_parser().parse_args(argv[1:])
            train, render, _, _, optimizer, _ = runner.create_nerf(args)
            self.assertEqual(train['network_fn'].key_frame_times.numel(), 3)
            self.assertEqual(len(optimizer.param_groups), 3)
            render.update(near=1., far=3.)
            origins = torch.zeros(4, 3)
            directions = torch.tensor([[0., 0., -1.]]).expand(4, 3)
            rgb, disp, _, _ = runner.render(8, 8, 8., rays=torch.stack((origins, directions)),
                                            frame_time=torch.tensor(.4), **render)
            self.assertTrue(torch.isfinite(rgb).all() and torch.isfinite(disp).all())
            # 执行真实 render_only 路径，只跳过不属于模型的 MP4 编码。
            # Run the real rendering entry point; bypass only the external video encoder.
            with patch.object(sys, 'argv', argv + ['--render_only', '--render_test']), patch.object(runner.imageio, 'mimwrite'):
                runner.train()
            rendered = next((root / 'logs' / 'smoke').glob('renderonly_test_*'))
            self.assertEqual(json.loads((rendered / 'frame_indices.json').read_text()), [1])
            self.assertTrue((rendered / 'estim' / '000.rgb.png').exists())
            self.assertTrue(np.isfinite(np.load(rendered / 'estim' / '000.depth.npy')).all())
            # 用同一数据执行 EndoNeRF 基线的一步训练。 / Exercise the legacy training path too.
            baseline = [v for v in argv if v not in ('--use_key_frame', '--use_std_as_depth_tolerance')]
            baseline[baseline.index('dnerfplus')] = 'direct_temporal'
            baseline[baseline.index('smoke')] = 'baseline'
            with patch.object(sys, 'argv', baseline):
                runner.train()
            self.assertTrue((root / 'logs' / 'baseline' / '000001.tar').exists())


if __name__ == '__main__':
    unittest.main()
