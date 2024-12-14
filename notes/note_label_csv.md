## csv class defination 


````
0: Correct prediction for class n and y (label n, y predicted as correspounding n, y).
1: Misclassified y as n (label y predicted as n).
2: Misclassified n as y (label n predicted as y).
````

````angular2html
if predict_cla == actual_label:
    result_value = 0  # Correct prediction
elif actual_label == 1 and predict_cla == 0:
    result_value = 1  # `y` predicted as `n`
elif actual_label == 0 and predict_cla == 1:
    result_value = 2  # `n` predicted as `y`

````

## jpg formation
red: Grayscale Image: Use one channel of the image as the normalized grayscale matrix.

green: Sine Channel: Add a channel where each pixel is filled with a value $V=\sin \left(\frac{\text { CSV value }}{\text { total categories }}\right)$

null(blue): Zero Channel: Add a third channel filled entirely with zeros.
Final RGB Image: Stack the grayscale, sine, and zero channels.



````angular2html
grayscale_channel = grayscale_image
                sine_value = np.sin(csv_value / num_categories)
                sine_channel = np.full(grayscale_channel.shape, sine_value, dtype=np.float32)
                zero_channel = np.zeros_like(grayscale_channel, dtype=np.float32)

                # Stack the channels
                stacked_image = np.stack([grayscale_channel, sine_channel, zero_channel], axis=-1)

````
## sinv_4c define

````
nn: Correct prediction for class n (CSV value = 0).
yy: Correct prediction for class y (CSV value = 0).
yn: Misclassified y as n (CSV value = 1).
ny: Misclassified n as y (CSV value = 2).
````