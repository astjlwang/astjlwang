import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pandas as pd
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import requests
from PIL import Image
from io import BytesIO

# ===================== 参数配置 =====================
ofile = 'allsky_src.fits'          # 全天空表格文件
catalog_file = 'final_catalog.txt' # 目标目录（包含 objname, ra, dec, log_mhalo）
radius_deg = 0.4                   # 视场半径
nbins = 10                         # 每个方向划分的格子数
use_log = True                     # 计数和曝光图使用对数色标
max_cos_dec = 0.1                  # 避免 cos(dec) 过小
output_dir = './grid_plots'        # 输出目录
ps1_output_dir = './ps1_plots'     # PanSTARRS 光学图输出目录

# ===================== 1. 读取全天空数据 =====================
print("读取全天空数据...")
maps = fits.open(ofile)[1].data

ra = maps['RA']
dec = maps['Dec']
exposure = maps['exposure']

channels = ['0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', '1.0-2.0']
counts = np.sum([maps[ch] for ch in channels], axis=0)

print(f"数据点数: {len(ra)}")
print("数据读取完成。")

# ===================== 2. 读取目标目录并按 log_mhalo 排序 =====================
cat = pd.read_csv(catalog_file, sep='\t')

# 按 log_mhalo 升序排序（小质量在前，大质量在后）
cat = cat.sort_values(by='log_mhalo', ascending=True).reset_index(drop=True)

obj_names = cat['objname'].values
ra_centers = cat['ra'].values
dec_centers = cat['dec'].values
log_mhalo = cat['log_mhalo'].values

n_objs = len(obj_names)
print(f"共 {n_objs} 个目标（已按 log_mhalo 升序排列）。")

# ===================== 3. 自适应网格化函数 =====================
def bin_region_adaptive(ra_center, dec_center, radius_deg,
                        ra_all, dec_all, counts_all, exposure_all, nbins,
                        max_cos_dec=0.1):
    rad_phys = np.radians(radius_deg)
    cosd = max(np.cos(np.radians(dec_center)), max_cos_dec)

    dec_min = dec_center - radius_deg
    dec_max = dec_center + radius_deg
    dec_edges = np.linspace(dec_min, dec_max, nbins + 1)

    phys_edges = np.linspace(-rad_phys, rad_phys, nbins + 1)
    ra_edges = ra_center + np.degrees(phys_edges) / cosd
    ra_min = ra_edges[0]
    ra_max = ra_edges[-1]

    wrap = False
    ra_all_adj = ra_all.copy()
    if ra_min < 0 or ra_max > 360:
        wrap = True
        ra_edges = ra_edges % 360
        sort_idx = np.argsort(ra_edges)
        ra_edges = ra_edges[sort_idx]
        mask_wrap = ra_all_adj < ra_center
        ra_all_adj[mask_wrap] += 360

    mask = (ra_all_adj >= ra_edges[0]) & (ra_all_adj <= ra_edges[-1]) & \
           (dec_all >= dec_min) & (dec_all <= dec_max)

    ra_region = ra_all_adj[mask]
    dec_region = dec_all[mask]
    counts_region = counts_all[mask]
    exposure_region = exposure_all[mask]

    if len(ra_region) == 0:
        return np.zeros((nbins, nbins)), np.zeros((nbins, nbins)), np.zeros((nbins, nbins))

    counts_grid, _, _ = np.histogram2d(dec_region, ra_region, bins=[dec_edges, ra_edges],
                                       weights=counts_region)
    exp_grid, _, _ = np.histogram2d(dec_region, ra_region, bins=[dec_edges, ra_edges],
                                    weights=exposure_region)

    rate_grid = np.zeros((nbins, nbins))
    mask_exp = exp_grid > 0
    rate_grid[mask_exp] = counts_grid[mask_exp] / exp_grid[mask_exp]

    return counts_grid, exp_grid, rate_grid

# ===================== 4. 为所有目标生成网格数据 =====================
counts_list = []
exp_list = []
rate_list = []

for idx, (name, ra_c, dec_c) in enumerate(zip(obj_names, ra_centers, dec_centers)):
    print(f"处理第 {idx+1}/{n_objs}: {name} (log_mhalo={log_mhalo[idx]:.2f})")
    c, e, r = bin_region_adaptive(ra_c, dec_c, radius_deg, ra, dec, counts, exposure, nbins, max_cos_dec)
    counts_list.append(c)
    exp_list.append(e)
    rate_list.append(r)

print("所有目标网格数据生成完成。")

# ===================== 5. PanSTARRS PS1 辅助函数 =====================
def ps1_getimages(ra, dec, filters="grizy"):
    """查询 PS1 可用图像列表"""
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table

def ps1_geturl(ra, dec, size=240, output_size=None, filters="grizy",
               fmt="jpg", color=False):
    """构建 PS1 cutout URL"""
    if color and fmt == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if fmt not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = ps1_getimages(ra, dec, filters=filters)
    if len(table) == 0:
        return None
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={fmt}")
    if output_size:
        url = url + f"&output_size={output_size}"
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            table = table[[0, len(table) // 2, len(table) - 1]]
        elif len(table) < 3:
            return None
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + f"&{param}={table['filename'][i]}"
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase + filename)
    return url

def ps1_getcolorim(ra, dec, size=240, output_size=None, filters="grizy", fmt="jpg"):
    """获取 PS1 彩色 cutout 图像"""
    url = ps1_geturl(ra, dec, size=size, filters=filters,
                     output_size=output_size, fmt=fmt, color=True)
    if url is None:
        return None
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    im = Image.open(BytesIO(r.content))
    return im

def fetch_ps1_cutout(ra, dec, fov_deg=0.8, output_size=480, filters="grizy"):
    """
    获取 PanSTARRS PS1 彩色 cutout。
    fov_deg 是视场直径（度），PS1 像素 = 0.25 角秒。
    PanSTARRS 覆盖 dec > -30°，超出范围返回 None。
    """
    if dec < -30.0:
        print(f"  PS1 跳过: dec={dec:.2f} < -30° (PS1 不覆盖)")
        return None
    size_pix = int(fov_deg * 3600 / 0.25)
    try:
        im = ps1_getcolorim(ra, dec, size=size_pix, output_size=output_size,
                            filters=filters, fmt="jpg")
        return im
    except Exception as e:
        print(f"  PS1 获取失败: {e}")
        return None

# ===================== 6. 获取 PanSTARRS 光学 cutout =====================
print("\n正在获取 PanSTARRS PS1 光学 cutout...")
ps1_images = []
ps1_valid = []

for idx, (name, ra_c, dec_c) in enumerate(zip(obj_names, ra_centers, dec_centers)):
    print(f"PS1 第 {idx+1}/{n_objs}: {name} (RA={ra_c:.4f}, Dec={dec_c:.4f})")
    im = fetch_ps1_cutout(ra_c, dec_c, fov_deg=2*radius_deg,
                          output_size=480, filters="grizy")
    ps1_images.append(im)
    ps1_valid.append(im is not None)

n_ps1_valid = sum(ps1_valid)
print(f"PS1 cutout 获取完成：{n_ps1_valid}/{n_objs} 个有数据。")

# ===================== 7. 绘图 =====================
ncols = max(int(np.ceil(np.sqrt(n_objs))) - 5, 1)
nrows = int(np.ceil(n_objs / ncols))

extent = [-radius_deg, radius_deg, -radius_deg, radius_deg]
cmap_counts = 'viridis'
cmap_exp = 'plasma'
cmap_rate = 'inferno'

# ---------- 7a. Counts 大图 ----------
fig_counts, axes_counts = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
fig_counts.suptitle('Counts (0.4–2.0 keV) per physical cell', fontsize=16)
if nrows * ncols == 1:
    axes_counts = [axes_counts]
else:
    axes_counts = axes_counts.flatten()

# ---------- 7b. Exposure 大图 ----------
fig_exp, axes_exp = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
fig_exp.suptitle('Exposure (s) per physical cell', fontsize=16)
if nrows * ncols == 1:
    axes_exp = [axes_exp]
else:
    axes_exp = axes_exp.flatten()

# ---------- 7c. Rate 大图 ----------
fig_rate, axes_rate = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
fig_rate.suptitle('Rate (counts/s) per physical cell', fontsize=16)
if nrows * ncols == 1:
    axes_rate = [axes_rate]
else:
    axes_rate = axes_rate.flatten()

for i in range(n_objs):
    counts_data = counts_list[i].copy()
    exp_data = exp_list[i].copy()
    rate_data = rate_list[i].copy()

    counts_data[counts_data == 0] = np.nan
    exp_data[exp_data == 0] = np.nan
    rate_data[rate_data == 0] = np.nan

    has_counts = not np.all(np.isnan(counts_data))
    has_exp = not np.all(np.isnan(exp_data))
    has_rate = not np.all(np.isnan(rate_data))

    label = f"{obj_names[i]}\nlog M$_{{halo}}$={log_mhalo[i]:.2f}"

    # ---- Counts ----
    ax_c = axes_counts[i]
    if use_log and has_counts:
        pos_vals = counts_data[counts_data > 0]
        min_pos = np.nanmin(pos_vals) if len(pos_vals) > 0 else 1e-10
        im_c = ax_c.imshow(counts_data, origin='lower', cmap=cmap_counts,
                           extent=extent, norm=LogNorm(vmin=min_pos, clip=True))
    elif has_counts:
        im_c = ax_c.imshow(counts_data, origin='lower', cmap=cmap_counts, extent=extent)
    else:
        im_c = ax_c.imshow(np.zeros((nbins, nbins)), origin='lower', cmap=cmap_counts, extent=extent)
    ax_c.set_title(label, fontsize=7)
    ax_c.set_xlabel('RA offset (deg)')
    ax_c.set_ylabel('Dec offset (deg)')
    if has_counts:
        divider = make_axes_locatable(ax_c)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_counts.colorbar(im_c, cax=cax)

    # ---- Exposure ----
    ax_e = axes_exp[i]
    if use_log and has_exp:
        pos_vals = exp_data[exp_data > 0]
        min_pos = np.nanmin(pos_vals) if len(pos_vals) > 0 else 1e-10
        im_e = ax_e.imshow(exp_data, origin='lower', cmap=cmap_exp,
                           extent=extent, norm=LogNorm(vmin=min_pos, clip=True))
    elif has_exp:
        im_e = ax_e.imshow(exp_data, origin='lower', cmap=cmap_exp, extent=extent)
    else:
        im_e = ax_e.imshow(np.zeros((nbins, nbins)), origin='lower', cmap=cmap_exp, extent=extent)
    ax_e.set_title(label, fontsize=7)
    ax_e.set_xlabel('RA offset (deg)')
    ax_e.set_ylabel('Dec offset (deg)')
    if has_exp:
        divider = make_axes_locatable(ax_e)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_exp.colorbar(im_e, cax=cax)

    # ---- Rate (改动1: 全部使用 LogNorm) ----
    ax_r = axes_rate[i]
    if has_rate:
        pos_vals = rate_data[rate_data > 0]
        min_pos = np.nanmin(pos_vals) if len(pos_vals) > 0 else 1e-10
        im_r = ax_r.imshow(rate_data, origin='lower', cmap=cmap_rate,
                           extent=extent, norm=LogNorm(vmin=min_pos, clip=True))
    else:
        im_r = ax_r.imshow(np.zeros((nbins, nbins)), origin='lower', cmap=cmap_rate, extent=extent)
    ax_r.set_title(label, fontsize=7)
    ax_r.set_xlabel('RA offset (deg)')
    ax_r.set_ylabel('Dec offset (deg)')
    if has_rate:
        divider = make_axes_locatable(ax_r)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_rate.colorbar(im_r, cax=cax)

for j in range(n_objs, nrows * ncols):
    axes_counts[j].axis('off')
    axes_exp[j].axis('off')
    axes_rate[j].axis('off')

fig_counts.tight_layout()
fig_exp.tight_layout()
fig_rate.tight_layout()
plt.show()

# ---------- 7d. PanSTARRS 光学大图（仅有数据的目标） ----------
if n_ps1_valid > 0:
    ncols_ps1 = max(int(np.ceil(np.sqrt(n_ps1_valid))) - 5, 1)
    if ncols_ps1 < 1:
        ncols_ps1 = 1
    nrows_ps1 = int(np.ceil(n_ps1_valid / ncols_ps1))

    fig_ps1, axes_ps1 = plt.subplots(nrows_ps1, ncols_ps1,
                                     figsize=(ncols_ps1 * 3.5, nrows_ps1 * 3))
    fig_ps1.suptitle('PanSTARRS PS1 Optical (grizy color)', fontsize=16)
    if nrows_ps1 * ncols_ps1 == 1:
        axes_ps1 = [axes_ps1]
    else:
        axes_ps1 = np.array(axes_ps1).flatten()

    plot_idx = 0
    for i in range(n_objs):
        if not ps1_valid[i]:
            continue
        ax = axes_ps1[plot_idx]
        ax.imshow(np.array(ps1_images[i]), origin='upper')
        label = f"{obj_names[i]}\nlog M$_{{halo}}$={log_mhalo[i]:.2f}"
        ax.set_title(label, fontsize=7)
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')
        plot_idx += 1

    for j in range(plot_idx, nrows_ps1 * ncols_ps1):
        axes_ps1[j].axis('off')

    fig_ps1.tight_layout()
    plt.show()
else:
    print("无可用的 PanSTARRS 数据，跳过光学图。")

# ===================== 8. 单独保存每个目标的 rate 子图 (LogNorm) =====================
print("正在单独保存每个目标的 rate 子图...")
os.makedirs(output_dir, exist_ok=True)

for i, name in enumerate(obj_names):
    rate_data = rate_list[i].copy()
    rate_data[rate_data == 0] = np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    has_rate = not np.all(np.isnan(rate_data))
    if has_rate:
        pos_vals = rate_data[rate_data > 0]
        min_pos = np.nanmin(pos_vals) if len(pos_vals) > 0 else 1e-10
        im = ax.imshow(rate_data, origin='lower', cmap='inferno',
                       extent=extent, norm=LogNorm(vmin=min_pos, clip=True))
    else:
        im = ax.imshow(np.zeros((nbins, nbins)), origin='lower', cmap='inferno', extent=extent)
    ax.set_title(f"{name}  (log M$_{{halo}}$={log_mhalo[i]:.2f})", fontsize=12)
    ax.set_xlabel('RA offset (deg)')
    ax.set_ylabel('Dec offset (deg)')
    plt.colorbar(im, ax=ax, label='Rate (counts/s)')

    safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    filename = os.path.join(output_dir, f"{safe_name}_rate.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {filename}")

# ===================== 9. 单独保存 PanSTARRS cutout =====================
print("正在单独保存 PanSTARRS cutout...")
os.makedirs(ps1_output_dir, exist_ok=True)

for i, name in enumerate(obj_names):
    if not ps1_valid[i]:
        continue
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.array(ps1_images[i]), origin='upper')
    ax.set_title(f"{name}  PS1 grizy (log M$_{{halo}}$={log_mhalo[i]:.2f})", fontsize=11)
    ax.set_xlabel('pixel')
    ax.set_ylabel('pixel')

    safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    filename = os.path.join(ps1_output_dir, f"{safe_name}_ps1.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {filename}")

print("全部完成！")
