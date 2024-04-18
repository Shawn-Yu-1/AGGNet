import numpy as np
from scipy import linalg
import torch
import torchvision as tv
import torchvision.transforms as TF
from tqdm import tqdm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(feature_real, feature_pre):
    """Calculation of the statistics used by the FID.
    """
    
    mu1 = np.mean(feature_real, axis=0)
    sigma1 = np.cov(feature_real, rowvar=False)
    mu2 = np.mean(feature_pre, axis=0)
    sigma2 = np.cov(feature_pre, rowvar=False)
    return mu1, sigma1, mu2, sigma2 


def get_feature_dataset(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset =tv.datasets.ImageFolder(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    real_arr = np.empty((len(files), dims))
    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            real = model(batch)[0]
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        real = real.squeeze(3).squeeze(2).cpu().numpy()
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        real_arr[start_idx:start_idx + pred.shape[0]] = real
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return real, pred_arr

def get_feature_generator(files, model, generator,dataset, batch_szie=4, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    real_arr = np.empty((len(files), dims))
    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            real = model(batch["image"])[0]
            pred = model(generator(batch))[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        real = real.squeeze(3).squeeze(2).cpu().numpy()
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        real_arr[start_idx:start_idx + pred.shape[0]] = real
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return real_arr, pred_arr


def get_feature_images(model, img_real, img_pre):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- img_real    : the ground true image
    -- img_pre     : the synthesis image
    
    """
    
    # model = model.cpu()
    # print(img_real.size())
    with torch.no_grad():
        real = model(img_real, return_features=True)
        pred = model(img_pre, return_features=True)
        # print(real.shape)
    # print(real.size())
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    real = real.cpu().numpy()
    pred = pred.cpu().numpy()

    return real, pred

def get_feature_images_incep(model, img_real, img_pre):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- img_real    : the ground true image
    -- img_pre     : the synthesis image
    
    """
    
    # model = model.cpu()
    # print(img_real.size())
    with torch.no_grad():
        real = model(img_real)
        pred = model(img_pre)
        # print(real.shape)
    # print(real.size())
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    real = real.cpu().numpy()
    pred = pred.cpu().numpy()

    return real, pred


def cal_fid(real, pred):
    mu1, sigma1, mu2, sigma2 = calculate_activation_statistics(real, pred)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
    return fid