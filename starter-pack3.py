# required packages
import os
import argparse
from utils import validation_reg

try:
    from osgeo import gdal
    from osgeo import osr, ogr
except:
    import gdal
    import osr

from utils import feature_engineering
from rf_regression2 import rf_regressor

if __name__ == "__main__":

 #   parser = argparse.ArgumentParser()
 #   parser.add_argument(
 #       "--data_path_So2Sat_pop_part1",
 #       type=str, help="Enter the path to So2Sat POP Part1 folder", required=True)
    
 #   parser.add_argument(
 #       "--data_path_So2Sat_pop_part2", required=True,
 #       type=str, help="Enter the path to So2Sat POP Part2 folder")
    
  #  args = parser.parse_args()
    
  #  all_patches_mixed_part1 = args.data_path_So2Sat_pop_part1
  #  all_patches_mixed_part2 = args.data_path_So2Sat_pop_part2

  #  all_patches_mixed_train_part1 = os.path.join(all_patches_mixed_part1, 'train')   # path to train folder
  #  all_patches_mixed_test_part1 = os.path.join(all_patches_mixed_part1, 'test')  # path to test folder
    
  #  all_patches_mixed_train_part2 = os.path.join(all_patches_mixed_part2, 'train')   # path to train folder
  #  all_patches_mixed_test_part2 = os.path.join(all_patches_mixed_part2, 'test')   # path to test folder

  #  print('\nPath to So2Sat POP Part1: ', all_patches_mixed_part1)
  #  print('Path to So2Sat POP Part2: ', all_patches_mixed_part2)
    
    # create features for training and testing data from So2Sat POP Part1 and So2Sat POP Part2
    feature_folder = feature_engineering('/tmp/data/So2Sat_POP_Part1/')
    feature_folder = feature_engineering('/tmp/data/So2Sat_POP_Part2/')

    # Perform regression, ground truth is population count (POP)
    prediction_csv = rf_regressor('/home/haicore-project-fzj/fzj_al.bazarova/countmein_submission/So2Sat_POP_features/')

    #validation_csv_path = prediction_csv.replace('prediction', 'validation')
    #validation_reg(prediction_csv, validation_csv_path, '/p/home/jusers/bazarova1/juwels/hai_countmein/countmein_submission/So2Sat_POP_features/test/')


