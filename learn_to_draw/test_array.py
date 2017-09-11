
import numpy as np

bbox = np.load('data/bbox1.npy')
final_mask = np.load('data/final_mask1.npy')
gt_label = np.load('data/gt_label1.npy')
image = np.load('data/image1.npy')
label = np.load('data/label1.npy')
prob = np.load('data/prob1.npy')
gt_mask = np.load('data/gt_mask1.npy')

save_array = []
save_array.append(bbox)
save_array.append(final_mask)
save_array.append(gt_label)
save_array.append(image)
save_array.append(label)
save_array.append(prob)
save_array.append(gt_mask)

save_array = np.asarray(save_array)
np.save("Array.npy",save_array)
save_array = np.load("Array.npy")

bbox = save_array[0]
final_mask = save_array[1]
gt_label = save_array[2]
image = save_array[3]
label = save_array[4]
prob = save_array[5]
gt_mask = save_array[6]
