"""PNG 底部边距标尺布局测试（比例以 Velox 参考 TIFF 实测标定 + 单位进位）。"""
import numpy as np
from PIL import Image

from velox_file_analyzer2 import (_scalebar_layout, _normalize_scalebar_unit,
                                  save_image_as_png, line_profile_ylabel,
                                  draw_line_profiles)


def test_line_profile_legend_clears_right_axis_ticks():
    """图例外置右移后，必须位于右轴(twinx)刻度数字的右侧，不得重叠。"""
    data = {
        'Al': {'profile_avg': np.linspace(0.1, 0.6, 80),
               'color': {'red': 1.0, 'green': 0.0, 'blue': 0.0}},
        'HAADF': {'profile_avg': np.linspace(1234.0, 98765.0, 80),
                  'color': {'red': 1.0, 'green': 1.0, 'blue': 1.0}},  # 右轴宽数字
    }
    fig = draw_line_profiles(data, output_path=None, pixel_size=0.861,
                             pixel_unit='nm', line_length_px=200,
                             quantification_mode='AtomicFraction')
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_left, ax_right = fig.axes[0], fig.axes[1]
    leg_box = ax_left.get_legend().get_window_extent(r)
    tick_right = max(t.get_window_extent(r).x1 for t in ax_right.get_yticklabels()
                     if t.get_text())
    assert leg_box.x0 >= tick_right + 4
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_line_profile_ylabel_by_mode():
    assert line_profile_ylabel('AtomicFraction') == 'Atomic fraction (%)'
    assert line_profile_ylabel('WeightFraction') == 'Weight fraction (%)'
    assert line_profile_ylabel('NetIntensity') == 'Net intensity (counts)'
    assert line_profile_ylabel('Intensity') == 'Intensity (counts)'
    assert line_profile_ylabel('') == 'Intensity of Elements'
    assert line_profile_ylabel(None) == 'Intensity of Elements'


def test_layout_proportional_matches_reference():
    # 0754 参考实测：3758px 图 → 文字墨迹≈91px(2.42%)、条厚≈19px(0.5%)
    fs, th, area = _scalebar_layout(3758, 1134, dpi=300)
    assert 36 <= fs <= 40          # ≈38pt → 墨迹≈91px（参考 2.42% 图高）
    assert 16 <= th <= 22          # ≈19px
    assert 0.04 * 3758 < area < 0.10 * 3758   # 边距区 ≈7%，远小于旧版 19%


def test_layout_small_image_shrinks_font():
    # 小图字号随比例缩小（下限 7pt），且边距区必须容纳字体 em 盒
    fs, th, area = _scalebar_layout(512, 136, dpi=300)
    assert fs == 7.0               # 0.0101*512=5.2 → 触底 7pt
    assert 3 <= th <= 24
    text_box_px = fs * 300 / 72.0
    assert area > text_box_px + th  # 字体盒+条 必须小于边距区（防溢出）


def test_layout_area_fits_font_all_sizes():
    for h in (64, 128, 256, 512, 1024, 4096):
        fs, th, area = _scalebar_layout(h, max(1, h // 4), dpi=300)
        assert area >= fs * 300 / 72.0 + th, (h, fs, th, area)


def test_normalize_units():
    assert _normalize_scalebar_unit(1000, 'nm') == (1.0, 'µm')
    assert _normalize_scalebar_unit(5000, 'nm') == (5.0, 'µm')
    assert _normalize_scalebar_unit(200, 'nm') == (200, 'nm')   # <1000 不进位
    assert _normalize_scalebar_unit(1500, 'nm') == (1500.0, 'nm')  # 非整千不进位
    assert _normalize_scalebar_unit(1000, 'um') == (1.0, 'mm')  # 别名归一后进位
    assert _normalize_scalebar_unit(1000000, 'nm') == (1.0, 'mm')  # 链式两级
    assert _normalize_scalebar_unit(0.5, 'nm') == (0.5, 'nm')
    assert _normalize_scalebar_unit(30, '1/nm') == (30, '1/nm')  # 倒数空间原样


def test_png_margin_is_proportional(tmp_path):
    """端到端：PNG 总高 = 图像高 + ~6-8% 边距（旧版 +19% 过深）。"""
    img = (np.random.default_rng(3).integers(0, 4000, size=(1024, 1024))
           .astype(np.uint16))
    out = tmp_path / 'sb.png'
    assert save_image_as_png(img, str(out), pixel_size=0.5, pixel_unit='nm',
                             display_range=(0, 4000), add_scalebar=True)
    with Image.open(out) as im:
        w, h = im.size
    assert abs(w - 1024) <= 4
    margin = h - 1024
    assert 0.02 * 1024 < margin < 0.12 * 1024


def test_small_png_text_stays_in_margin(tmp_path):
    """纯白底图端到端：任何黑色标尺内容都不得越入图像区（文字溢出回归）。"""
    for side in (256, 512):
        img = np.full((side, side), 60000, dtype=np.uint16)  # 归一化后全白
        out = tmp_path / f'sb_{side}.png'
        assert save_image_as_png(img, str(out), pixel_size=2.0, pixel_unit='nm',
                                 display_range=(0, 60000), add_scalebar=True)
        a = np.array(Image.open(out).convert('L')).astype(np.float64)
        dark = a < 128
        assert dark[:side, :].sum() == 0, f"{side}px: 标尺内容画进了图像区"
        assert dark[side:, :].sum() > 100, f"{side}px: 边距区没有标尺内容"
