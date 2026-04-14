from pathlib import Path
import streamlit as st
import numpy as np
import imageio
from io import BytesIO
from PIL import Image
import rasterio
from rasterio.io import MemoryFile

from utils import ensure_rgb, vegetation_mask, water_mask, reliefs_mask, cities_mask, visualize_masks, compute_stats


st.set_page_config(layout='wide', page_title='Analizador de Imagen Satelital')


def load_image_from_bytes(b: bytes):
    try:
        arr = imageio.v2.imread(BytesIO(b))
        return arr
    except Exception:
        try:
            img = Image.open(BytesIO(b)).convert('RGB')
            return np.array(img)
        except Exception:
            # try rasterio for GeoTIFF bytes
            try:
                with MemoryFile(b) as mem:
                    with mem.open() as src:
                        data = src.read()
                        # data shape: (bands, rows, cols)
                        if data.ndim == 3:
                            # take first 3 bands or stack
                            bands = data[:3]
                            arr = np.transpose(bands, (1, 2, 0))
                            return arr
                        else:
                            return data
            except Exception:
                return None


def get_default_image():
    # try to find a sensible image in data/
    data_dir = Path('data')
    if data_dir.exists():
        allowed = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        for f in sorted(data_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in allowed:
                continue
            # prefer rasterio for tiff/geotiff
            try:
                if f.suffix.lower() in ('.tif', '.tiff'):
                    with rasterio.open(f) as src:
                        data = src.read()
                        # data shape: (bands, rows, cols)
                        if data.ndim == 3:
                            bands = data[:3]
                            arr = np.transpose(bands, (1, 2, 0))
                            return arr
                        else:
                            # fallback: squeeze and return
                            return np.squeeze(data)
                else:
                    arr = imageio.v2.imread(f)
                    return arr
            except Exception:
                try:
                    img = Image.open(f).convert('RGB')
                    return np.array(img)
                except Exception:
                    continue
    return None

def main():
    st.sidebar.title('Controles')
    uploaded = st.sidebar.file_uploader('Sube una imagen (JPG/PNG/TIF)', type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
    selections = st.sidebar.multiselect('Detectar en imagen', ['Vegetación', 'Agua', 'Relieves', 'Ciudades'], default=['Vegetación', 'Agua'])
    veg_thresh = st.sidebar.slider('Umbral Vegetación (ExG)', 0, 100, 20)
    relief_thresh = st.sidebar.slider('Umbral Relieves (edge)', 0, 255, 30)

    if uploaded is not None:
        b = uploaded.read()
        img = load_image_from_bytes(b)
        source_label = f'Upload: {uploaded.name}'
    else:
        img = get_default_image()
        source_label = 'Imagen por defecto (data/)'

    if img is None:
        st.warning('No se pudo leer la imagen. Sube una imagen válida.')
        return

    img_rgb = ensure_rgb(img)

    tabs = st.tabs(['Original', 'Detecciones', 'Estadísticas'])

    with tabs[0]:
        st.header('Imagen Original')
        # st.write(source_label)
        st.image(img_rgb, width=1200)

    masks = {}
    if 'Vegetación' in selections:
        masks['Vegetación'] = vegetation_mask(img_rgb, thresh=veg_thresh)
    if 'Agua' in selections:
        masks['Agua'] = water_mask(img_rgb)
    if 'Relieves' in selections:
        masks['Relieves'] = reliefs_mask(img_rgb, thresh=relief_thresh)
    if 'Ciudades' in selections:
        masks['Ciudades'] = cities_mask(img_rgb)

    with tabs[1]:
        st.header('Detecciones')
        if masks:
            overlay = visualize_masks(img_rgb, masks, alpha=0.5)
            st.image(overlay, width=800)
            # show individual masks
            cols = st.columns(2)
            for i, (name, m) in enumerate(masks.items()):
                with cols[i % 2]:
                    st.subheader(name)
                    st.image((m * 255).astype('uint8'), clamp=True, width=300)
        else:
            st.info('Selecciona algún tipo de detección en el sidebar.')

    with tabs[2]:
        st.header('Estadísticas')
        if masks:
            stats = compute_stats(masks)
            for k, v in stats.items():
                st.write(f'- **{k}**: {v:.2f}% del área total')
        else:
            st.info('Sin detecciones para mostrar estadísticas.')


if __name__ == '__main__':
    main()
