import pandas as pd
import numpy as np
import os
import cv2
import h5py

#Turn gaze vector to visual angle
def tranform_to_visual_angle(gazeX, gazeY, gazeZ):
  #x,y,z = normalize_gaze_vector(gaze_vector)
  theta = np.arctan2(gazeX,gazeZ)
  phi = np.arctan2(gazeY,gazeZ)
  #turn to degrees
  theta = np.rad2deg(theta)
  phi = np.rad2deg(phi)

  return theta,phi


def normalize_visual_angle(column):
  min_val = column.min()
  max_val = column.max()
  return (column - min_val) / (max_val - min_val)


def calculate_outliers_from_column(column):
  # IQR
  Q1 = np.percentile(column, 25, method='midpoint')
  Q3 = np.percentile(column, 75, method='midpoint')
  IQR = Q3 - Q1
  #print("Q1: ", Q1, " Q3: ", Q3, " IQR: ", IQR)

  # Calculate the upper and lower limits
  upper = Q3 + 1.5 * IQR
  lower = Q1 - 1.5 * IQR
  #print("Upper Bound:", upper, " Lower Bound:", lower)

  #Find Outlier values below and above thresholds (indexes)
  upper_array = np.where(column >= upper)[0]
  lower_array = np.where(column <= lower)[0]
  outliers = np.hstack((upper_array,lower_array))
  #print("Outliers:" , outliers.shape)
  #print(outliers)
  #test = [0,1,3]
  #test = np.array(test)
  return outliers


def remove_outliers(dtframe):
  gazex = dtframe['angleX']
  gazey = dtframe['angleY']

  outliersX = calculate_outliers_from_column(gazex)
  outliersY = calculate_outliers_from_column(gazey)
  
  print("Outliers in angleX:", len(outliersX))
  print("Outliers in angleY:", len(outliersY))


  full_outliers = np.hstack((outliersX, outliersY))
  full_outliers = np.unique(full_outliers)
  print("Full number of  unique outliers:", full_outliers.shape[0])


  dtframe.drop(dtframe.index[full_outliers], inplace=True)
  return dtframe 






def correct_path(image_name):
  full_path = os.path.join(pathToImageFolder,image_name)
  return full_path


def load_norm_images(image_path):
  try:
    img = cv2.imread(image_path)
    img = np.asarray(img) / 255.0
    return img
  except Exception as e:
    print(f"Error loading image {image_path}: {e}")
    return None



def load_images_in_batches(dtframe, batch_size, limit, starting_point):
  num_batches = int(np.ceil(len(dtframe) / batch_size)) #calculate how many  total batches
  start_batch = int(np.floor(starting_point / batch_size))  # start from the correct batch
  processed_count = starting_point
  for i in range(start_batch, num_batches):
    batch = df.iloc[i * batch_size: (i+1) * batch_size]
    indexes_list = []
    
    for idx, row in batch.iterrows():
      image = load_norm_images(row['images_color'])
      if image is None:
         dtframe.iloc[idx,3] = np.nan
         dtframe.dropna(inplace = True)

      processed_count+=1
      indexes_list.append(idx)
      if processed_count >= limit:
        break
    #yield np.array(batch_of_images), np.array(indexes_list)
    yield np.array(indexes_list)
    if processed_count >= limit:
      break




def split_list(my_list, seq_length):
    true_seq = seq_length + 1
    #Split the list into sublists of length true_seq
    sublists = [my_list[i:i + true_seq] for i in range(0, len(my_list), true_seq)]
    
    #Check the last sublist and remove if not perfect division
    if len(sublists[-1]) < true_seq:
        #print("List to be discarded:", sublists[-1])
        sublists.pop()  # Remove the last sublist if it's not of the desired length
    return sublists








# Function to create or extend the HDF5 dataset in a specified file
def append_to_hdf5(file_name, data, dataset_name, shape):
    with h5py.File(file_name, 'a') as f:
        if dataset_name not in f:
            # Create dataset if it doesn't exist
            f.create_dataset(dataset_name, data=data, maxshape=(None, *shape), chunks=True)
        else:
            # Resize and append data to the existing dataset
            dataset = f[dataset_name]
            dataset.resize(dataset.shape[0] +  1, axis=0)
            dataset[-1] = data



################################################################################
pathDataset= r'C:\Users\knina\Desktop\Thesis Stuff\Datasets\openneeds.parquet'
print("Reading dataset...............\n")
df= pd.read_parquet(pathDataset)

print("Cleaning dataset..............\n")
#Drop rows with NaN & keep only useful data
df.dropna(inplace=True)

#Reset duplicates indices
duplicated_indices = df.index[df.index.duplicated()] 
df.reset_index(drop=True, inplace=True)

#Keep useful columns only
df = df[['subject_id','time', 'scene_index','images_color', 'gaze_x', 'gaze_y', 'gaze_z']]


################################################################################
                            #Preprocess Gaze Information

print("Transforming gaze vectors to visual angles.......\n")
df['angleX'], df['angleY'] = tranform_to_visual_angle(df['gaze_x'], df['gaze_y'], df['gaze_z'])


#Remove outliers
print("Removing outliers.............")
print("Shape before removing outliers: ", df.shape)
#df = remove_outliers(df)

print("Shape after removing outliers: ", df.shape)

#Calculate min and max values before normalization (necessary for denormalization of predictions)
Xmin = df['angleX'].min()
Xmax = df['angleX'].max()

Ymin = df['angleY'].min()
Ymax = df['angleY'].max()

print("Angle X min:", Xmin, "max:", Xmax)
print("Angle Y min:", Ymin, "max:", Ymax)



#normalization
print("\nNormalizing gaze..............\n")
df['angleXNorm'] = normalize_visual_angle(df['angleX'])
df['angleYNorm'] = normalize_visual_angle(df['angleY'])


 
################################################################################
                            #Preprocess Frames

pathToImageFolder = r"D:\Kwn\OPENNEEDS_dataset\images_colorUNzipped\images_color"


print("Getting correct image paths.........")
df['images_color'] = df['images_color'].apply(correct_path)


#x min -44.87077192237714 max: 48.61006921041047
#Angle Y min: -50.34826187331623 max: 39.331806514990646

batch_size=1001          #how many images per batch
starting=899099
limit = 1300000             #how many TOTALLY (to split into batches....)
sequence_length = 10
h,w,c = 71,128,3

#filename = r"C:\Users\knina\Desktop\Thesis Stuff\Datasets\test.hdf5"
filename = r"D:\Kwn\OPENNEEDS_dataset\final_withOutliers3.hdf5"

#399999
#init file (run only one time)
with h5py.File(filename, 'w') as f:
  f.create_dataset('X1_dataset', shape=(0, sequence_length, h, w, c), maxshape=(None, sequence_length, h, w, c), chunks=True)
  f.create_dataset('X2_dataset', shape=(0, sequence_length, 2), maxshape=(None, sequence_length, 2), chunks=True)
  f.create_dataset('Y_dataset', shape=(0,2), maxshape=(None, 2), chunks=True)


image_generator = load_images_in_batches(df,batch_size, limit, starting)


total_sequences = 0
batch_counter = 0
for indices in image_generator: #se auto to simeio exw batch_size arithmo eikonwn kai ta swsta indices
   batch_counter+=1
   #print("\nBatch no.",batch_counter, "with indices:", indices)
   print("\nBatch no.",batch_counter, "with indices:", indices[0],"-",indices[-1])

   #for each batch separate to sequences!!! make sure seq are of same user and scene!!
   #total_indices = indices.shape[0]
   #split_into = np.floor(total_indices / sequence_length)
   #print(total_indices,"indices to be split into", split_into, "sequences, with sequence_length:", sequence_length)
   #split_indices = np.array_split(indices, split_into)
   split_indices = split_list(indices,sequence_length)

   for miniList in split_indices: #an exw xwrisei se 2 listes, tha treksei dio fores. minilist = [0,1,2,3,4], [5,6,7,8,9]
      starting_index = miniList[0]
      #print(f"Starting index:",starting_index)
      X1, X2, Y = [], [], []
      row = df.iloc[starting_index]
      success = 0
      
      iter=0
      for element in miniList:
         nextRow = df.iloc[element]
         if nextRow['subject_id'] == row['subject_id'] and nextRow['scene_index'] == row['scene_index']:
            success+=1
            if iter <= sequence_length -1:
              #print(element, "add to sequence")
              temp1  = load_norm_images(nextRow['images_color'])
              X1.append(temp1)
              x,y = nextRow[['angleXNorm','angleYNorm']].values
              gp = (x,y)
              X2.append(gp)
            else:
               #print(element, "add to label y")
               x,y =  nextRow[['angleXNorm','angleYNorm']].values
               gp = (x,y)
               Y.append(gp)
         else:
            #print("Rejected due to element: ", element)
            break
         
         if success == sequence_length +1 :
            #print("Sequence ready: ", miniList)
            total_sequences +=1
            X1 = np.array(X1)
            X2 = np.array(X2)
            Y = np.array(Y)
            #print("X1: ",X1.shape, "X2: ", X2.shape, "Y: ", Y.shape)

            append_to_hdf5(filename, X1, 'X1_dataset', (sequence_length,h,w,c))
            append_to_hdf5(filename, X2, 'X2_dataset', (sequence_length,2))
            append_to_hdf5(filename, Y,  'Y_dataset',  (1,2))
         iter+=1

            

#print("finito")
print("\nTotal sequences created:", total_sequences)








      





