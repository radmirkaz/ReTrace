import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm.notebook import tqdm
import ipywidgets
import torch
from sklearn.preprocessing import normalize


def denormilize_img(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def plot_imgs(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(
        nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(num_rows * 16, num_cols * 16)
    )

    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            if row_title is not None:
                ax.set(xlabel=row_title[col_idx])

    plt.tight_layout()


def plot_img(imgs, row_title=None, **imshow_kwargs):
    plot_imgs([imgs], row_title, **imshow_kwargs)


def apply_mask(image, mask, threshold=0.3, color=(1, 0, 0), alpha=0.7):
    image = image.copy()
    rmask = mask > threshold
    for c in range(3):
        image[:, :, c] = np.where(
            rmask, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c]
        )
    return image


color_map = np.array(plt.get_cmap("plasma").colors)


def apply_depth_map(frame, coords, depth):
    global color_map
    depth_img = np.zeros_like(frame)
    depth = (np.clip(depth, 0.0, 120.0) / 120 * 255.0).astype(np.long)
    for (x, y), d in zip(coords, depth):
        depth_img = cv2.circle(depth_img, (int(x), int(y)), 7, color_map[d], -1)

    dmask = depth_img > 0.01

    frame = frame.copy()
    frame[dmask] = depth_img[dmask] * 0.4 + frame[dmask] * 0.6
    return frame


def plot_imgs_with_masks(imgs, masks, threshold=0.3, row_title=None, **imshow_kwargs):
    new_imgs = [apply_mask(img, mask, threshold) for img, mask in zip(imgs, masks)]
    plot_imgs(new_imgs, row_title, **imshow_kwargs)


def plot_img_with_mask(img, mask, threshold=0.3, row_title=None, **imshow_kwargs):
    plot_imgs_with_masks([img], [mask], threshold, row_title, **imshow_kwargs)


def calc_dist(embedding1, embedding2):
    embedding1 = normalize(embedding1)
    embedding2 = normalize(embedding2)
    return np.linalg.norm(embedding1 - embedding2)


def plot_model_preds_interact(model, ds, transform, default_value=0):
    def plot_model_preds(index):
        data = ds.get_triplet_images(index)
        with torch.no_grad():
            outputs = [model(img.unsqueeze(0).to(model.device)) for img, _ in data]
        labels = [torch.argmax(l, axis=1) for l, f in outputs]

        imgs = [transform.denormalize(image=d[0])["image"] for d in data]
        embs = [o[1].detach().cpu().numpy() for o in outputs]
        titles = ["[Target: %s, Pred: %s]" % (int(d[1]), int(p)) for d, p in zip(data, labels)]

        titles[1] += " | Positive: %.2f" % calc_dist(embs[0], embs[1])
        titles[2] += " | Negative: %.2f" % calc_dist(embs[0], embs[2])

        plot_imgs(imgs, row_title=titles)

    ipywidgets.interact(
        plot_model_preds, index=ipywidgets.IntText(min=0, max=len(ds), step=1, value=default_value)
    )
