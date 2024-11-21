import torch
from torchvision.utils import save_image

class FourierDiscriminantMeta(object):
    def __init__(self, shift: bool = True, one_side: bool = False):
        """
        Initialize the FourierDiscriminantMeta class.

        Parameters:
        shift (bool): Whether to shift the zero-frequency component to the center.
        one_side (bool): Whether to compute the one-sided FFT.
        """
        self.shift = shift
        self.one_side = one_side
    
    def get_meta(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fourier transform of the input image.

        Parameters:
        img (torch.Tensor): 4-D tensor (b,1,h,w) representing the input image.

        Returns:
        torch.Tensor: Fourier transformed image.
        """
        if self.one_side:
            f = torch.fft.rfft2(img, dim=(-2, -1))
        else:
            f = torch.fft.fft2(img, dim=(-2, -1))
        
        if self.shift:
            f = torch.fft.fftshift(f, dim=(-2, -1))
        
        return f
    
    def visualize(self, f: torch.Tensor, save_path: str = None, binarization: bool = False, normalize: bool = False) -> torch.Tensor:
        """
        Visualize the Fourier transformed image.

        Parameters:
        f (torch.Tensor): Fourier transformed image.
        save_path (str): Path to save the visualized image.
        binarization (bool): Whether to binarize the image.
        normalize (bool): Whether to normalize the image.

        Returns:
        torch.Tensor: Visualized image.
        """
        f_img = 20 * torch.log(torch.abs(f))
        if binarization:
            f_img = torch.where(f_img > 0, torch.ones_like(f_img), torch.zeros_like(f_img))
            
        if save_path is not None:
            save_image(f_img, save_path, normalize=normalize)
        
        return f_img
    
    def inverse_meta(self, f: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse Fourier transform of the input image.

        Parameters:
        f (torch.Tensor): 4-D tensor (b,1,h,w) representing the Fourier transformed image.

        Returns:
        torch.Tensor: Inverse Fourier transformed image.
        """
        if self.shift:
            f = torch.fft.ifftshift(f, dim=(-2, -1))
            
        if self.one_side:
            f = torch.fft.irfft2(f, dim=(-2, -1))
        else:
            f = torch.fft.ifft2(f, dim=(-2, -1))
        
        return torch.abs(f)
    
    def rgb2gray(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert an RGB tensor image to grayscale.

        Parameters:
        tensor (torch.Tensor): 4-D tensor (b,c,h,w) representing the RGB image.

        Returns:
        torch.Tensor: 4-D tensor (b,1,h,w) representing the grayscale image.
        """
        gray = 0.299 * tensor[:, 0, :, :] + 0.587 * tensor[:, 1, :, :] + 0.114 * tensor[:, 2, :, :]
        return gray.unsqueeze(1)