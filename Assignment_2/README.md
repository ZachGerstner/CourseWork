Patch Notes For Gaussian Blur Code
Generated : 12 Feb 2019
Last Touch : 13 Feb 2019
===================================
12 Feb
Main.cpp
+Implemented memcpy to device 
+Implemented malloc for devie pointers
+Freed malloced memory
*Device pointers for blurred are not init?*
gaussain.cu
+created index
+created loop for x and y
+implemented curr row and cols
+adds to pix val
+writes out to d_out
*no idea if the alg is right.*
*Issues with the recombine function*
->Not sure how to get the char * to go to uchar4
+Implemented unify uchar solution
*Ran into issue with libraries...
**Meeting with the David to troubleshoot*

