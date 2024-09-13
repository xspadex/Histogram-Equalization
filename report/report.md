# Revamping Images: Advanced Histogram Equalization Techniques

## Abstract
Histogram Equalization (HE) is a commonly used technique for image enhancement that improves contrast by adjusting the grayscale distribution of an image, making details in low-contrast images clearer. However, while enhancing contrast, the HE algorithm may also introduce issues such as noise, color distortion, and excessive brightness enhancement. To address these shortcomings, this study explores several optimization methods based on the HE algorithm, including Bi-Histogram Equalization (BBHE), Dynamic Bi-Histogram Equalization (DSIHE), Weighted Guided Image Filtering (WGIF), and Entropy-based Weighted Histogram Equalization (EIHE). Through experiments, we performed both quantitative and qualitative analyses of the effects of different optimization methods, aiming to improve image contrast and detail clarity while preserving original color information and reducing noise and other distortions. The experimental results demonstrate that each optimization method achieves significant improvements in different scenarios, with WGIF and EIHE showing excellent performance in retaining image features and reducing noise. All relevant code can be found in the repository: https://github.com/xspadex/Histogram-Equalization.git. The contributions of each group member can be found at the end of the article in 6. Contribution Statement.

## 1. Introduction
Image enhancement techniques play a crucial role in computer vision and digital image processing. **Histogram Equalization (HE)** is a classical method for enhancing image contrast and improving overall visual quality. However, HE often causes significant changes in the mean brightness, especially when dealing with very bright or very dark images, which may result in the loss of image features and color distortion. To address these issues, we adopted several improved algorithms to optimize the results by limiting brightness changes, preserving local details, and controlling contrast enhancement.

First, we employed **Brightness Preserving Bi-Histogram Equalization (BBHE)**, which divides the histogram into two sub-histograms based on the mean brightness of the original image and equalizes them separately, effectively maintaining the average brightness of the image. Unlike traditional HE, BBHE can suppress brightness changes while enhancing image contrast to a certain extent. However, although BBHE preserves some brightness information, its performance in very bright or very dark images is still inadequate, especially in terms of edge details.

To solve this problem, we introduced **Dual Sub-Image Histogram Equalization (DSIHE)**. Unlike BBHE, DSIHE divides the histogram based on the median brightness, allowing for more balanced processing of both light and dark regions of the image. However, in practical image processing, we found that BBHE and DSIHE still result in blurred details or suboptimal contrast when dealing with high-brightness or low-brightness images. Therefore, we adopted **Minimum Mean Brightness Error Bi-Histogram Equalization (MMBEBHE)**, which divides the histogram more accurately using the Minimum Absolute Mean Brightness Error (MABE), further improving the enhancement effect, particularly in terms of detail preservation and brightness balance.

Although these bi-histogram-based methods improve image quality to some extent, global equalization methods can still lead to over-enhancement in local areas, especially in regions with high contrast. To address this, we introduced **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**, which processes the image in small blocks to avoid distortion caused by global equalization. CLAHE limits the histogram in each block to prevent excessive contrast enhancement, better preserving local details.

Additionally, to tackle issues such as edge information loss and noise amplification, we applied **Weighted Guided Image Filtering (WGIF)**, an edge-preserving filtering technique that smooths the image while maintaining edge sharpness. WGIF introduces edge-aware weights, allowing the filter to perform better in complex regions of the image, thus enhancing detail preservation.

Despite WGIF's excellent performance in detail preservation, we found that it still results in the loss of edge details in some scenarios. To further optimize image enhancement, we combined **CLAHE** with **WGIF**. First, CLAHE is applied to the reflectance component of the image to enhance edge information, followed by WGIF to maintain the image's average brightness and local details. This improved method enhances image contrast while reducing edge blurring and noise amplification issues.

Through these improvements and integrations, we successfully overcame many limitations of traditional histogram equalization, particularly in enhancing details, preserving brightness, and suppressing noise. Ultimately, our improved method significantly enhances image quality in complex scenarios, providing more natural enhancement effects for images with varying brightness and contrast.

## 2. Original HE Algorithm
#### 2.1 Principle

Histogram Equalization (HE) is an image enhancement technique that improves contrast by adjusting the distribution of pixel intensity values. The mathematical principle of HE involves the following steps:

1. **Histogram Generation**:
   A histogram shows the frequency of each pixel intensity level in the image. For a typical image, the intensity levels range from 0 to 255. Each bar in the histogram represents the number of pixels corresponding to a particular intensity level.

2. **Cumulative Distribution Function (CDF) Calculation**:
   The cumulative distribution function is derived from the histogram, showing the cumulative frequency distribution. The CDF describes how pixel intensity values are distributed in the image. The CDF is calculated using the following formula:
   $$
   CDF(k) = \frac{\sum_{j=0}^{k} n(j)}{N}
   $$
   
   where n(j) is the number of pixels with intensity level j, N is the total number of pixels in the image (equal to W*H), and k is the current intensity level.

3. **Intensity Mapping**:
   Based on the CDF, the pixel intensity values are mapped to new values. The mapping formula is:
$$
s(k) = \text{round} \left((L-1)\sum_{j=0}^{k} p(j) \right)
$$

​	where kk is the original intensity level, ss is the new intensity level, and LL is the number of intensity 	levels.

HE adjusts the intensity distribution to make the pixel values more uniformly distributed, enhancing contrast in the image. However, HE can sometimes introduce color distortion and noise amplification.

#### 2.2 Experimental Effect Analysis
Through experiments, we found that Histogram Equalization (HE) does indeed have a certain effect on image enhancement. The effect can be significantly observed through histograms, which show that it moves the gray levels from the edges towards the center in some way. This mapping method is quite simple and direct, but also relatively coarse. It works well for images with clear contrast, but for images that are particularly bright or dark, it doesn’t produce good results. This can be seen from the two images below.

//TODO put images and histograms here @Jiang Changjiu

## 3. Advanced Algorithms
### 3.1 BBHE && DSIHE && MMBEBHE
According to the images processed by HE algorithm, the algorithm significantly changed the average brightness of the image, especially when the image is bright or dark. This may bring about two issues. On one hand, the image may lose characteristics since the change of saturation masks or blurs details like texture and color gradation. On the other hand, the colors of the image may be distorted, causing an unnatural effect.

To solve this problem, we considered adopting a series of bi-histogram based algorithm. First we applied brightness preserving bi-histogram equalization (BBHE) algorithm proposed by Kim. The algorithm divides the input histogram into two sub-histograms according to the average brightness of the original image, and the two sub-histograms are then histogram equalized separately. To maintain the average brightness, the equalization range is limited to the original image brightness range. We also implemented dualistic sub-image histogram equalization (DSIHE), which is also a bi-histogram base algorithm. Different from BBHE, it uses median brightness as the basis of division.

According to the image processing results, the processing effects of BBHE and DSIHE on images with high and low mean brightness are still not ideal. Then we apply minimum mean brightness error bi-histogram equalization (MMBEBHE) algorithm. This algorithm traverses all possible brightness values and gets the sub-histogram through minimum absolute mean brightness error (MABE). Consequently, the algorithm properly enhances the images.

### 3.2 DPHE && BHE2PL

Double Plateaus Histogram Equalization (DPHE) is an advanced image enhancement algorithm that improves upon traditional Histogram Equalization by addressing the issue of over-enhancement and noise amplification commonly found in HE. The DPHE algorithm divides the original image's histogram into two parts, setting two distinct plateau thresholds to control the stretching of pixel intensities. This dual-limiting mechanism ensures that extreme intensity values are capped, thus reducing the amplification of noise and preserving more natural image details.

By setting plateau limits, DPHE prevents the excessive stretching of high-intensity regions, which is a common issue in traditional HE. DPHE maintains more subtle image features and textures, provides better control over the contrast enhancement, resulting in more visually appealing images, especially in scenarios with complex lighting conditions.

Bi-Histogram Equalization with Dual Plateau Limits (BHE2PL) is an advanced histogram equalization algorithm designed to improve contrast enhancement by splitting the image's histogram into two sub-histograms and applying independent dual plateau limits (thresholds) to each. This method helps control contrast enhancement more effectively and prevents over-enhancement and noise amplification, which are common issues in traditional Histogram Equalization.

By dividing the histogram into two parts, BHE2PL allows for separate handling of darker and brighter regions, enhancing contrast in both areas without overwhelming one another. The use of dual plateau limits in both sub-histograms helps control the stretching of grayscale values, preventing extreme contrast enhancements that are often observed in traditional HE. Clipping the histogram at dual plateau limits reduces noise amplification, particularly in low-contrast or darker regions of the image.


### 3.3 WGIF
Based on WGIF, Mu, Q et al. proposed an enhancement method. In the proposed method, WGIF is applied to estimate the illumination component. Both the guide and input images are the intensity image $S_I$ , and the output is the estimated illumination component, denoted $S_{IL}$. The brightness of the estimated illumination component $S_{IL}$ is often very low, so the proposed method uses adaptive gamma correction and then get $S_{ILG}$: $S_{ILG}(x,y)=S_{IL}(x,y)^{\phi(x,y)}, \phi(x,y) = \frac{S_{IL}(x,y)+a}{1+a}, a=1-\frac{1}{mn}\sum_{x=1}^m\sum_{y=1}^n S_{IL}(x,y)$

where $S_{ILG}(x,y)$ is the corrected illumination component, and $m$, $n$ are the height, width of the original image, respectively.

After adaptive gamma correction, the dynamic range of the image is compressed. Thus the corrected illumination component $S_{ILG}$ is stretched linearly to obtain the result image $S_{ILGf}$ .

The reflection component $S_{IR}$ can be obtained from: $S_{IR}(x,y)=S_{I}(x,y)/S_{IL}(x,y)$.

Since noise mainly exists in the reflection component, $S_{IR}$ is then processed by WGIF, giving the denoised reflection component $S_{IRH}$. Then, the processed illumination component $S_{ILGf}$ is multiplied by the denoised reflection component $S_{IRH}$ to give the fused intensity image $S_{IE}$. Finally, the S-hyperbolic tangent function is used to improve the brightness of the fused image $S+{IE}$, and the enhanced intensity image $S_{IEf}$ is obtained: $S_{IEf}(x,y)=\frac{1}{1+\exp{(-8(S_{IE}-b))}}, b=\frac{1}{mn}\sum_{x=1}^{m}\sum_{y=1}^{n}S_{IE}(x,y)$, where b is the mean intensity of $S_{IE}$, and $m$, $n$ are the height and width of $S_{IE}$, respectively.

### 3.4 EIHE
To address the drawbacks of HE (Histogram Equalization) in losing edge details near boundaries and amplifying noise, we adopted an approach that uses an edge intensity histogram instead of a simple brightness histogram, as proposed in [1], to enhance the contrast of the image.

Traditional HE uses luminance histograms, which represent the number of pixels at each brightness level (0-255) without considering the spatial relationships or local features of the pixels. For each brightness level m, E(m) denotes the number of pixels with brightness m, implying that each pixel contributes equally to the histogram. However, the edge intensity histogram represents the overall brightness difference between each pixel and its eight neighboring pixels at each brightness level. The calculation process is as follows:
$$
E(m) = \sum_{n=0}^{L-1}{C(m,n) \times  \left| m - n \right|}\quad m = 0, 1, \cdots, L-1
$$


Here m is the brightness level of the current pixel, C(m,n) is the number of eight-neighborhood pixels with brightness level n, and L is the total number of brightness levels. For each pixel, its edge intensity value is accumulated into the histogram bin corresponding to its brightness level. Thus, each pixel's contribution to the histogram varies depending on its edge intensity. This edge intensity-based histogram focuses more on edge information. Equalization processing based on this histogram can better capture structural information in the image, effectively preserving and enhancing image details. 

To address the potential issue of noise amplification in the aforementioned method, according to the article, we decompose the image into a base layer and a detailed layer. We apply edge intensity histogram equalization to the base layer, while performing a linear transformation on the detailed layer, and finally sum those two layers.

During the process of processing sample images, we found that this method did not produce satisfactory results for overly bright or dark images. Therefore, we introduced a parameter alpha when calculating the transformation function for edge intensity. After obtaining the CDF function F, we apply `F_power = F ** alpha`. When alpha < 1, it enhances the contrast in dark regions, and when alpha > 1, it enhances the contrast in bright regions. Additionally, by calculating the average brightness value of the image, we determine whether the current image is too bright or too dark, and assign corresponding values to alpha for processing. This approach results in better processing effects for both bright and dark images.

## 4. Experiment
### 4.1 定量分析
提供实验的数值结果，展示不同方法在对比度、亮度分布、颜色保真度等方面的表现。通过表格或图示展示不同算法的对比效果。

### 4.2 定性分析
基于视觉观察评价各算法在实际场景中的表现，探讨不同方法在处理图像细节、颜色保留和噪声抑制时的优缺点。

### 4.3 Discussion
讨论实验结果背后的原因，分析各优化方法在不同应用场景中的适用性，并指出现有方法的局限性。

## 5. Conclusion
For the cat image shown in Figure 3a, the colors are relatively light and the contrast
is low. From its brightness histogram, Figure 4a, it can be seen that the grayscale values
are concentrated in the higher range, making the image appear overly bright with a loss
of details. The visual effect is poor, requiring an overall enhancement of contrast and
edge details. These enhancement methods generally lowered the image brightness, and
improved the image contrast and clarity, making contours and details more prominent.
Visually, HE significantly increased contrast but severely distorted colors and amplified noise. WGIF achieved a more uniform brightness distribution but lost some original
color information. MMBEBHE and DPHE also have issues with the loss of original color
information. BBHE, DSIHE, and BHE2PL retained some of the original tones, but details in the brightest areas were still lost. EIHE preserved most of the original features
but generated more noise, making the image look unnatural. CLAHE produced a relatively good effect in this image, enhancing image details and contrast while preserving
the natural appearance and color information of the original image.

For the indoor scene shown in Figure 13a, the original image appears generally dark,
with details obscured in shadows. Its brightness histogram, Figure 14a, shows that the
grayscale values are concentrated in the low range, resulting in low contrast with much
information hidden in dark areas. All methods improved the brightness in dark places to
varying degrees, making details in shadow areas more visible. For the basic HE method,
which amplifies noise and causes overexposure in some areas (such as the window) due
to excessive enhancement, other optimization methods also show various improvements.
However, issues like loss of original colors and insufficient brightness enhancement still
exist in methods like BBHE, DSIHE, and BHE2PL. The best visual results are achieved
by WGIF and EIHE, which make the brightness distribution more uniform while retaining
some original features.

## 6. Contribution Statement

The following list is in no particular order. Each section of the article was written by the student responsible for its implementation or analysation.

**Jiang Changjiu(G2402840A)** served as group leader, coordinating the team's efforts to ensure the project progressed on schedule. He was also responsible for the implementation of the original HE algorithm, the implementation of visualization tools, and managed the overall structure and formatting of the report.

**Miao Kehao()** //TODO

**Qiu Zhiheng(G2303454H)** responsible for the theoretical analysis and implementation of the BBHE, DSIHE, and MMBEBHE algorithms, as well as part of the experimental results analysis and report writing.

**Shen Tongfei()** //TODO

**Zuo Yuqi()** //TODO

## References
Li, Z., Zheng, J., Zhu, Z., Yao, W., & Wu, S. (2014). Weighted guided image filtering. *IEEE Transactions on Image processing*, *24*(1), 120-129.

Mu, Q., Wang, X., Wei, Y., & Li, Z. (2021). Low and non-uniform illumination color image enhancement using weighted guided image filtering. *Computational Visual Media*, *7*, 529-546.

Kim, T. S., & Kim, S. H. (2023). An improved contrast enhancement for dark images with non-uniform illumination based on edge preservation. *Multimedia Systems*, *29*(3), 1117-1130. [1]

Aquino-Morínigo P B, Lugo-Solís F R, Pinto-Roa D P, et al. Bi-histogram equalization using two plateau limits[J]. Signal, Image and Video Processing, 2017, 11: 857-864.

Liang K, Ma Y, Xie Y, et al. A new adaptive contrast enhancement algorithm for infrared images based on double plateaus histogram equalization[J]. Infrared Physics & Technology, 2012, 55(4): 309-315.

Kim, Y. T. (1997). Contrast enhancement using brightness preserving bi-histogram equalization.  IEEE Transactions on Consumer Electronics, 43(1), 1–8

Wang, Y., Chen, Q., & Zhang, B. (1999). Image enhancement based on equal area dualistic subimage histogram equalization method. IEEE Transactions on Consumer Electronics, 45(1), 68–75.

Chen, S.-D., & Ramli, A. R. (2003). Minimum mean brightness error bi-histogram equalization in contrast enhancement. IEEE Transactions on Consumer Electronics, 49(4), 1310–1319.
