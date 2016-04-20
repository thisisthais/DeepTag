import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

image_size = 640  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  for image_index, image in enumerate(image_files):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file, flatten=True).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        print(image_file)
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index + 1
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_folders = ['global_landmarks/bigben_train', 'global_landmarks/colosseum_train', 'global_landmarks/eiffeltower_train', 'global_landmarks/empirestatebuilding_train', 'global_landmarks/goldengatebridge_train', 'global_landmarks/hollywoodsign_train', 'global_landmarks/leaningtower_train', 'global_landmarks/londoneye_train', 'global_landmarks/notredamecathedral_train', 'global_landmarks/statueofliberty_train', 'global_landmarks/stpetersbasilica_train', 'global_landmarks/tokyotower_train',];
test_folders = ['global_landmarks/bigben_test', 'global_landmarks/colosseum_test', 'global_landmarks/eiffeltower_test', 'global_landmarks/empirestatebuilding_test', 'global_landmarks/goldengatebridge_test', 'global_landmarks/hollywoodsign_test', 'global_landmarks/leaningtower_test', 'global_landmarks/londoneye_test', 'global_landmarks/notredamecathedral_test', 'global_landmarks/statueofliberty_test', 'global_landmarks/stpetersbasilica_test', 'global_landmarks/tokyotower_test',];
train_datasets = maybe_pickle(train_folders, 400)
test_datasets = maybe_pickle(test_folders, 40)







