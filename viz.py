import matplotlib.pyplot as plt
import os, imageio
import numpy as np
import cv2
import utils

plt.rcParams['figure.figsize'] = (10.0, 8.0)  

# Data loader - already done for you
def get_image(size=512, \
              image_url='https://bmild.github.io/fourfeat/img/lion_orig.png'):

    # Download image, take a square crop from the center 
    img = imageio.imread(image_url)[..., :3] / 255.
    c = [img.shape[0]//2, img.shape[1]//2]
    r = 256
    img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]

    if size != 512:
        img = cv2.resize(img, (size, size))

    plt.imshow(img)
    plt.show()

    # Create input pixel coordinates in the unit square
    coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    x_test = np.stack(np.meshgrid(coords, coords), -1)
    test_data = [x_test, img]
    train_data = [x_test[::2, ::2], img[::2, ::2]]

    return train_data, test_data

def plot_training_curves(train_loss, train_psnr, test_psnr):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  
    # plot the training loss
    plt.subplot(2, 1, 1)
    plt.plot(train_loss)
    plt.title('MSE history')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')

    # plot the training and testing psnr
    plt.subplot(2, 1, 2)
    plt.plot(train_psnr, label='train')
    plt.plot(test_psnr, label='test')
    plt.title('PSNR history')
    plt.xlabel('Iteration')
    plt.ylabel('PSNR')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def plot_reconstruction(p, y):
    size = int(np.sqrt(p.shape[0]))
    p_im = p.reshape(size,size,3)
    y_im = y.reshape(size,size,3)

    plt.figure(figsize=(12,6))

    # plot the reconstruction of the image
    plt.subplot(1,2,1), plt.imshow(p_im), plt.title("reconstruction")

    # plot the ground truth image
    plt.subplot(1,2,2), plt.imshow(y_im), plt.title("ground truth")

    print("Final Test MSE", utils.mse(y, p))
    print("Final Test psnr", utils.psnr(y, p))

def plot_reconstruction_progress(predicted_images, y, N=8):
    size = int(np.sqrt(predicted_images.shape[1]))
    total = len(predicted_images)
    step = np.ceil(total/N).astype(int)
    plt.figure(figsize=(24, 4))

    # plot the progress of reconstructions
    for i, j in enumerate(range(0,total, step)):
        plt.subplot(1, N, i+1)
        plt.imshow(predicted_images[j].reshape(size,size,3))
        plt.axis("off")
        plt.title(f"iter {j}")

    # plot ground truth image
    plt.subplot(1, N+1, N+1)
    plt.imshow(y.reshape(size,size,3))
    plt.title('GT')
    plt.axis("off")
    plt.show()
    
def plot_feature_mapping_comparison(outputs, gt):
    # plot reconstruction images for each mapping
    plt.figure(figsize=(24, 4), dpi=200)
    N = len(outputs)
    for i, k in enumerate(outputs):
        plt.subplot(1, N+1, i+1)
        size = int(np.sqrt(outputs[k]['pred_imgs'][-1].shape[0]))
        plt.imshow(outputs[k]['pred_imgs'][-1].reshape(size, size, -1))
        plt.title(k)
    plt.subplot(1, N+1, N+1)
    plt.imshow(gt)
    plt.title('GT')
    plt.show()

    # plot train/test error curves for each mapping
    iters = len(outputs[k]['train_psnrs'])
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    for i, k in enumerate(outputs):
        plt.plot(range(iters), outputs[k]['train_psnrs'], label=k)
    plt.title('Train error')
    plt.ylabel('PSNR')
    plt.xlabel('Training iter')
    plt.legend()
    plt.subplot(122)
    for i, k in enumerate(outputs):
        plt.plot(range(iters), outputs[k]['test_psnrs'], label=k)
    plt.title('Test error')
    plt.ylabel('PSNR')
    plt.xlabel('Training iter')
    plt.legend()
    plt.savefig("lion_high_res.png", dpi=200)
    plt.show()

# Save out video
# def create_and_visualize_video(outputs, size=size, epochs=epochs, filename='training_convergence.mp4'):
def create_and_visualize_video(outputs, size=32, epochs=1000, filename='training_convergence.mp4'):
    all_preds = np.concatenate([outputs[n]['pred_imgs'].reshape(epochs,size,size,3)[::25] for n in outputs], axis=-2)
    data8 = (255*np.clip(all_preds, 0, 1)).astype(np.uint8)
    f = os.path.join(filename)
    imageio.mimwrite(f, data8, fps=20)

    # Display video inline
    from IPython.display import HTML
    from base64 import b64encode
    mp4 = open(f, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    N = len(outputs)
    if N == 1:
        return HTML(f'''
        <video width=256 controls autoplay loop>
              <source src="{data_url}" type="video/mp4">
        </video>
        ''')
    else:
        return HTML(f'''
        <video width=1000 controls autoplay loop>
              <source src="{data_url}" type="video/mp4">
        </video>
        <table width="1000" cellspacing="0" cellpadding="0">
          <tr>{''.join(N*[f'<td width="{1000//len(outputs)}"></td>'])}</tr>
          <tr>{''.join(N*['<td style="text-align:center">{}</td>'])}</tr>
        </table>
        '''.format(*list(outputs.keys())))
    