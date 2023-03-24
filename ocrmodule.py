import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import cv2,math
from PIL import Image
import torchvision.transforms as transforms

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)

        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        # print(np.array(img)) 
        # print(img)
        img.sub_(0.5).div_(0.5)
        return img
    
class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        images = batch

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors

class OCR_PILE:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        opt = parser.parse_args()
        opt.workers = 4
        opt.batch_size = 2
        opt.saved_model = "best_accuracy.pth"
        opt.batch_max_length = 10
        opt.imgH = 32
        opt.rgb = False
        opt.PAD = False
        opt.imgW = 200
        opt.character = "abcdefghklmnprstuvxyz0123456789"
        opt.Transformation = "None"
        opt.FeatureExtraction = "RCNN"
        opt.SequenceModeling = "None"
        opt.Prediction = "CTC"
        opt.num_fiducial = 20

        opt.input_channel = 1
        opt.output_channel = 512
        opt.hidden_size = 256

        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

        self.converter = CTCLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        if opt.rgb:
            opt.input_channel = 3
        print(opt)
        self.model_ocr = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
                opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
                opt.SequenceModeling, opt.Prediction)
        self.model_ocr = torch.nn.DataParallel(self.model_ocr).to(device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        self.model_ocr.load_state_dict(torch.load(opt.saved_model, map_location=device))
        self.model_ocr = self.model_ocr.module
        self.model_ocr.eval()
        img = torch.randn(2, 1, 32, 200, requires_grad=True).to(device)
        y = self.model_ocr(img)  # dry run

        self.AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        # predict
        self.model_ocr.eval()
        self.opt = opt

    def ocr(self,img_np):
        with torch.no_grad():
            
            img = Image.fromarray(img_np).convert('L')
            # img = img.crop((0, img.size[1]//2, img.size[0], img.size[1]))
            image_tensors = self.AlignCollate_demo([img])
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            preds = self.model_ocr(image)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)

            preds_str = self.converter.decode(preds_index, preds_size)

            # continue
            if 'Attn' in self.opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS] 
            return preds_str

if __name__ == '__main__':
    ocrpile = OCR_PILE()
    image_path_list = ""
    img_np = cv2.imread(image_path_list)
    result = ocrpile.ocr(img_np)[0]
    print(result)
