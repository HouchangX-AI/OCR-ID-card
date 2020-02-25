from keras import Input, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))

from PIL import Image, ImageDraw
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
import AdvancedEast.cfg as cfg
from AdvancedEast.utils.preprocess import resize_image
from AdvancedEast.utils.util import sigmoid,cut_text_line
from AdvancedEast.utils.nms import nms


class East:
    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(None, None, cfg.num_channels),
                               dtype='float32')
        # 用vgg16做提取图像特征工作
        vgg16 = VGG16(input_tensor=self.input_img,
                      weights='/Users/lige/公开课/一小时身份证识别/ocr-ID-card/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                      include_top=False)
        if cfg.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = [vgg16.get_layer('block%d_pool' % i).output
                  for i in cfg.feature_layers_range]
        self.f.insert(0, None)
        # diff=1
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num

    def g(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == cfg.feature_layers_num:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D((2, 2))(self.h(i))

    def h(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1,
                            activation='relu', padding='same',)(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3,
                            activation='relu', padding='same',)(bn2)
            return conv_3

    # 组装模型网络结构，返回model
    def east_network(self):
        before_output = self.g(cfg.feature_layers_num)
        inside_score = Conv2D(1, 1, padding='same', name='inside_score'
                              )(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code'
                             )(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord'
                              )(before_output)
        east_detect = Concatenate(axis=-1,
                                  name='east_detect')([inside_score,
                                                       side_v_code,
                                                       side_v_coord])
        
        return Model(inputs=self.input_img, outputs=east_detect)

    def predict_for_test(self, img_path, weight_path = cfg.saved_model_weights_file_path, threshold = cfg.pixel_threshold, quiet=False):
        '''
            预测测试用，每次预测单张图片，输出预测中间结果图，预测结果图，可选切分图片
            img_path:图片路径
            weght_path:模型权重路径
            threshold:判断结果阈值
            quiet:是否屏蔽调试输出，当是4个点的时候认为是文本框，不是的时候不输出但是会打印日志
        '''
        # 创建模型
        east_detect = self.east_network()
        # 加载模型权重
        east_detect.load_weights(weight_path)
        print ('load weight success')
        img = image.load_img(img_path)
        d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
        img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        img = image.img_to_array(img)
        #去均值中心化
        img = preprocess_input(img, mode='tf')
        x = np.expand_dims(img, axis=0)
        y = east_detect.predict(x)
        y = np.squeeze(y, axis=0)
        y[:, :, :3] = sigmoid(y[:, :, :3])
        # 对比概率大于阈值的元素
        cond = np.greater_equal(y[:, :, 0], threshold)
        activation_pixels = np.where(cond)
        quad_scores, quad_after_nms = nms(y, activation_pixels)
        with Image.open(img_path) as im:
            im_array = image.img_to_array(im.convert('RGB'))
            d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
            quad_im = im.copy()
            draw = ImageDraw.Draw(im)
            for i, j in zip(activation_pixels[0], activation_pixels[1]):
                px = (j + 0.5) * cfg.pixel_size
                py = (i + 0.5) * cfg.pixel_size
                line_width, line_color = 1, 'red'
                if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                    if y[i, j, 2] < cfg.trunc_threshold:
                        line_width, line_color = 2, 'yellow'
                    elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                        line_width, line_color = 2, 'green'
                draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                        (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                        (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                        (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                        (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                        width=line_width, fill=line_color)
            im.save(img_path + '_act.jpg')
            quad_draw = ImageDraw.Draw(quad_im)
            txt_items = []
            for score, geo, s in zip(quad_scores, quad_after_nms,
                                    range(len(quad_scores))):
                if np.amin(score) > 0:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='red')
                    if cfg.predict_cut_text_line:
                        cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                                    img_path, s, save_path=os.path.dirname(img_path))
                    rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                    rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                    txt_item = ','.join(map(str, rescaled_geo_list))
                    txt_items.append(txt_item + '\n')
                elif not quiet:
                    print('quad invalid with vertex num less then 4.')
            quad_im.save(img_path + '_predict.jpg')
            if cfg.predict_write2txt and len(txt_items) > 0:
                with open(img_path[:-4] + '.txt', 'w') as f_txt:
                    f_txt.writelines(txt_items)

    def predict_for_ocr(self, img_path, weight_path = cfg.saved_model_weights_file_path, threshold = cfg.pixel_threshold, quiet=True):
        # 创建模型
        east_detect = self.east_network()
        # 加载模型权重
        east_detect.load_weights(weight_path)
        print ('load weight success')
        img = image.load_img(img_path)
        d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
        img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        img = image.img_to_array(img)
        #去均值中心化
        img = preprocess_input(img, mode='tf')
        x = np.expand_dims(img, axis=0)
        y = east_detect.predict(x)
        y = np.squeeze(y, axis=0)
        y[:, :, :3] = sigmoid(y[:, :, :3])
        # 对比概率大于阈值的元素
        cond = np.greater_equal(y[:, :, 0], threshold)
        activation_pixels = np.where(cond)
        quad_scores, quad_after_nms = nms(y, activation_pixels)
        result = {}
        # tmp_img_name = []
        with Image.open(img_path) as im:
            im_array = image.img_to_array(im.convert('RGB'))
            # print('im_array.shape:', im_array.shape)
            d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')

            if cfg.generation_prediction_img_during_predict:
                quad_im = im.copy()
                draw = ImageDraw.Draw(im)
                for i, j in zip(activation_pixels[0], activation_pixels[1]):
                    px = (j + 0.5) * cfg.pixel_size
                    py = (i + 0.5) * cfg.pixel_size
                    line_width, line_color = 1, 'red'
                    if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                        if y[i, j, 2] < cfg.trunc_threshold:
                            line_width, line_color = 2, 'yellow'
                        elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                            line_width, line_color = 2, 'green'
                    draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                            (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                            (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                            (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                            (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                            width=line_width, fill=line_color)
                im.save(os.path.join(cfg.tmp_img_dir_name, os.path.basename(img_path) + '_act.jpg'))
                quad_draw = ImageDraw.Draw(quad_im)

            for score, geo, s in zip(quad_scores, quad_after_nms,
                                    range(len(quad_scores))):
                if np.amin(score) > 0:
                    if cfg.generation_prediction_img_during_predict:
                        quad_draw.line([tuple(geo[0]),
                                        tuple(geo[1]),
                                        tuple(geo[2]),
                                        tuple(geo[3]),
                                        tuple(geo[0])], width=2, fill='red')

                    min_xy, max_xy, img_name = cut_text_line(geo, scale_ratio_w, scale_ratio_h, 
                        im_array,img_path, s)
                    result[img_name] = {'coordinate' : [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]}
                    # tmp_img_name.append(img_name)
                elif not quiet:
                    print('quad invalid with vertex num less then 4.')
            quad_im.save(os.path.join(cfg.tmp_img_dir_name, os.path.basename(img_path) + '_predict.jpg'))
        return result
            

if __name__ == '__main__':
    east = East()
    # east_network = east.east_network()
    # east_network.summary()
    img_path = 'D:/AI_src/AdvancedEAST-master/demo/0001.jpg'
    # east.predict_for_test(img_path)
    files_name = east.predict_for_ocr(img_path)
    # print (files_name)