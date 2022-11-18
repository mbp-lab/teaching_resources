import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Helper Functions
def display_one_image(image, title, subplot, color='black', mask=None):
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image, )
    plt.title(title, fontsize=16)
    
def display_nine_images(images, titles, preds, start, title_colors=None):
    subplot = 331
    plt.figure(figsize=(13,13))
    for i in range(9):
        color = 'black' if title_colors is None else title_colors[i]
        idx = start+i
        display_one_image(images[idx], f'Actual={titles[idx]} \n Pred={preds[idx]} \n Index = {idx}', 331+i, color)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.show()
    
def setup_gridsearch(*args):
    """
    Usage:
    lrs = [0.001, 0.01, 0.1]
    kappas = [0.01, 0.1, 1, 10]
    betas = [0.001, 0.1, .9]

    grid = setup_gridsearch(lrs, kappas, betas)

    exps = []
    for lr, kappa, beta in (pbar := tqdm.tqdm(grid)):
        pbar.set_description(f'{lr=}, {kappa=}, {beta}')
    """
    return list(itertools.product(*args))

# def image_title(label, prediction):
#   # Both prediction (probabilities) and label (one-hot) are arrays with one item per class.
#     class_idx = np.argmax(label, axis=-1)
#     prediction_idx = np.argmax(prediction, axis=-1)
#     if class_idx == prediction_idx:
#         return f'{CLASS_LABELS[prediction_idx]} [correct]', 'black'
#     else:
#         return f'{CLASS_LABELS[prediction_idx]} [incorrect, should be {CLASS_LABELS[class_idx]}]', 'red'

# def get_titles(images, labels, model):
#     predictions = model.predict(images)
#     titles, colors = [], []
#     for label, prediction in zip(classes, predictions):
#         title, color = image_title(label, prediction)
#         titles.append(title)
#         colors.append(color)
#     return titles, colors