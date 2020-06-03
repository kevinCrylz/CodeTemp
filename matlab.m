//**  built-in math functions - Eg. pi, sin(), sqrt(), eig()  **//


//**  Manipulation  **//
clear                       // clear workspace variables
clc                         // clear console

save {file} {data}          // save workspace variable to file
load {file} 

//** Condition  **//
if x==1  ...  else  ...  end
for idx = 1:10  ...  end

//**  Vector  **//          // index starts at 1
x = [1 2 3; 4 5 6]  
x = 1:2:10                  // start value : step : end value (**inclusive**)
x = linspace(1,10,5)        // start value, end value, # of value

//**  Matrix  **//  
x = rand(n)                 // n-by-n uniformly-distributed random matrix
x = rand(x, y)              // x-by-y uniformly-distributed random matrix
x = randi([-5,5], x, y)     // x-by-y uniformly-distributed integer random matrix ranging from -5 to 5
x = randn(x, y)             // x-by-y normally-distributed random matrix
x = zeros(x, y)             // x-by-y zero matrix

cnt = numel(m)              // number of element
bool = v1 > v2              // element-wise logical operation
v(v>4) = 4

v = data(end-2:end, :)      // matrix value
[dr, dc] = size(matrix)     // Matrix dimension

matMul = x * y              // Matrix multiplication
eleMul = x .* y             // element-wise multiplication

max = max(v)                // max value of an array
[vMax, ivMax] = max(v)      // [max value, max index]
round = round(v)            // round off

//**  Plot  **// 
hold on                     // plot in same graph
hold off                    // clear graph and create new graph
close all                   // close windows

plot(v, 'LineWidth', 3)     // plot value against index
plot(x, y, 'ro-')           // plot y against x
loglog(x, y, '')            // plot in log scale

title('')                   // Chart title
legend('a','b','c')
ylabel('')                  // y-axis label

img = imread('')
imshow(img)
imwrite(img, '')

im2double(img)
imshowpair(img1, img2)