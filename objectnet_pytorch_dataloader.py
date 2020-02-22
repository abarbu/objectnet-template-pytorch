from torchvision.datasets.vision import VisionDataset
from PIL import Image
import glob

class ObjectNetDataset(VisionDataset):
    """
    ObjectNet dataset.

    Args:
        root (string): Root directory where images are downloaded to. The images can be grouped in folders. (the folder structure will be ignored)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, 'transforms.ToTensor'
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        img_format (string): jpg
                             png - the original ObjectNet images are in png format
    """

    def __init__(self, root, transform=None, target_transform=None, transforms=None, img_format="jpg"):
        """Init ObjectNet pytorch dataloader."""
        super(ObjectNetDataset, self).__init__(root, transforms, transform, target_transform)
        #from objectnet_competition_api import ObjectNet

        #self.objectnet = ObjectNet(root)
        #self.ids = list(sorted(self.objectnet.imgs.keys()))

        self.loader = self.pil_loader
        self.img_format = img_format
        files = glob.glob(root+"/**/*."+img_format)
        self.pathDict = {}
        for f in files:
            self.pathDict[f.split("/")[-1]] = f
        self.imgs = list(self.pathDict.keys())

    def __getitem__(self, index):
        """
        Get an image and its label.

        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by 'objectnet.loadAnns'.
        """
        img, target = self.getImage(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def getImage(self, index):
        """
        Load the image and its label.

        Args:
            index (int): Index
        Return:
            tuple: Tuple (image, target). target is the object returned by 'objectnet.loadAnns'.
        """
        img = self.loader(self.pathDict[self.imgs[index]])

        # crop out red border
        width, height = img.size
        cropArea = (2, 2, width-2, height-2)
        img = img.crop(cropArea)
        return (img, self.imgs[index])

    def __len__(self):
        """Get the number of ObjectNet images to load."""
        return len(self.imgs)

    def pil_loader(self, path):
        """Pil image loader."""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
