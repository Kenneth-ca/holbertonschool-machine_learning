# 0x01. Plotting

## Tasks

### 0. Line Graph
Complete the following source code to plot y as a line graph:

* y should be plotted as a solid red line
* The x-axis should range from 0 to 10

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/math/0x01-plotting/src/fig0.png)

### 1. Scatter
Complete the following source code to plot x ↦ y as a scatter plot:

* The x-axis should be labeled Height (in)
* The y-axis should be labeled Weight (lbs)
* The title should be Men's Height vs Weight
* The data should be plotted as magenta points

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/math/0x01-plotting/src/fig1.png)

### 2. Change of scale
Complete the following source code to plot x ↦ y as a line graph:

* The x-axis should be labeled Time (years)
* The y-axis should be labeled Fraction Remaining
* The title should be Exponential Decay of C-14
* The y-axis should be logarithmically scaled
* The x-axis should range from 0 to 28650

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/math/0x01-plotting/src/fig2.png)

### 3. Two is better than one
Complete the following source code to plot x ↦ y1 and x ↦ y2 as line graphs:

* The x-axis should be labeled Time (years)
* The y-axis should be labeled Fraction Remaining
* The title should be Exponential Decay of Radioactive Elements
* The x-axis should range from 0 to 20,000
* The y-axis should range from 0 to 1
* x ↦ y1 should be plotted with a dashed red line
* x ↦ y2 should be plotted with a solid green line
* A legend labeling x ↦ y1 as C-14 and x ↦ y2 as Ra-226 should be placed in the upper right hand corner of the plot

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/math/0x01-plotting/src/fig3.png)

### 4. Frequency
Complete the following source code to plot a histogram of student scores for a project:

* The x-axis should be labeled Grades
* The y-axis should be labeled Number of Students
* The x-axis should have bins every 10 units
* The title should be Project A
* The bars should be outlined in black

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/math/0x01-plotting/src/fig4.png)

### 5. All in One
Complete the following source code to plot all 5 previous graphs in one figure:

* All axis labels and plot titles should have a font size of x-small (to fit nicely in one figure)
* The plots should make a 3 x 2 grid
* The last plot should take up two column widths (see below)
* The title of the figure should be All in One

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/math/0x01-plotting/src/fig5.png)

### 6. Stacking Bars
Complete the following source code to plot a stacked bar graph:

* fruit is a matrix representing the number of fruit various people possess
    * The columns of fruit represent the number of fruit Farrah, Fred, and Felicia have, respectively
    * The rows of fruit represent the number of apples, bananas, oranges, and peaches, respectively
* The bars should represent the number of fruit each person possesses:
    * The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
    * Each fruit should be represented by a specific color:
        * apples = red
        * bananas = yellow
        * oranges = orange (#ff8000)
        * peaches = peach (#ffe5b4)
        * A legend should be used to indicate which fruit is represented by each color
     * The bars should be stacked in the same order as the rows of fruit, from bottom to top
     * The bars should have a width of 0.5
* The y-axis should be labeled Quantity of Fruit
* The y-axis should range from 0 to 80 with ticks every 10 units
* The title should be Number of Fruit per Person

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/math/0x01-plotting/src/fig6.png)

### 7. Gradient
Complete the following source code to create a scatter plot of sampled elevations on a mountain:

* The x-axis should be labeled x coordinate (m)
* The y-axis should be labeled y coordinate (m)
* The title should be Mountain Elevation
* A colorbar should be used to display elevation
* The colorbar should be labeled elevation (m)

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/math/0x01-plotting/src/fig100.png)

### 8. PCA
Principle Component Analysis (PCA) is a vital procedure used in data science for reducing the dimensionality of data (in turn, decreasing computation cost). It is also largely used for visualizing high dimensional data in 2 or 3 dimensions. For this task, you will be visualizing the Iris flower data set . You will need to download the file pca.npz to test your code. You do not need to push this dataset to github. Complete the following source code to visualize the data in 3D:

* The title of the plot should be PCA of Iris Dataset
* pca_data is a np.ndarray of shape (150, 4)
    * 150 => the number of flowers
    * 4 => petal length, petal width, sepal length, sepal width
* labels is a np.ndarray of shape (150,) containing information about what species of iris each data point represents:
    * 0 => Iris Setosa
    * 1 => Iris Versicolor
    * 2 => Iris Virginica
* The columns of pca_data represent the 3 dimensions of the reduced data, i.e., x, y, and z, respectively
* The x, y, and z axes should be labeled U1, U2, and U3, respectively
* The data points should be colored based on its labels using the plasma color map

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/math/0x01-plotting/src/fig101.png)
