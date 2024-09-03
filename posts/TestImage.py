



import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import PiecewiseAffineTransform
from skimage.transform import AffineTransform
from skimage.io import imread


def ChooseTransform(transform,im_shape):
    rows, cols = im_shape[0], im_shape[1]

    src_cols = np.linspace(0, cols, 50)
    src_rows = np.linspace(0, rows, 50)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    dst_rows = src[:, 1] 
    dst_cols = src[:, 0]


    # add sinusoidal oscillation to row coordinates
    #transform='yscaleDarcy299Fig147'
    if transform=='identity':
        dst_rows = src[:, 1] 
        dst_cols = src[:, 0]
    elif transform=='yscale':
        yscaleparam=1.3
        dst_rows = dst_rows*yscaleparam
        dst_cols = src[:, 0]
    elif transform=='yscaleDarcy299Fig151':
        sc_factor=1.25
        k_x=sc_factor*((dst_cols-cols/2.0)/cols)**1.0+1.0
        dst_rows = k_x*(dst_rows-rows/2.0)+rows/2.0
        #dst_cols = src[:, 0]
    elif transform=='yscaleDarcy299Fig149':
        dst_rows = 0.8*(dst_rows-rows/2.0)*(2.5-1.4*((dst_cols-cols/2.0)/(cols/2.0))**2.0)+rows/2.0
        #dst_cols = (dst_cols-cols/2.0)*(dst_rows-rows/2.0)**2.0+cols/2.0

        #dst_cols = src[:, 0]  

        #dst_cols = src[:, 0]  
    elif transform=='yscaleDarcy299Fig147':
        k_y=-0.5
        dst_cols = dst_cols+k_y*dst_rows
        #dst_cols = src[:, 0]    
    elif transform=='yscalelog':
        L=np.max(dst_rows)
        dst_rows = (np.log(1.0+(dst_rows)/L))*L/np.log(2.0)
        #dst_cols = src[:, 0]    
    elif transform=='yscalepower':
        L=np.max(dst_rows)
        n=3
        dst_rows = L*(dst_rows/L)**n
        #dst_cols = src[:, 0]        
    elif transform=='radialscale':
        dst_rows_max=np.max(dst_rows)
        dst_cols_max=np.max(dst_cols)

        R_typical=30.0
        radius=R_typical*((radius.astype(float))/R_typical)**1.5
        #theta=theta*1.0
        dst_rows=radius*np.sin(theta)+centre[0]
        dst_cols=radius*np.cos(theta)+centre[1]

        #dst_rows=dst_rows/np.max(dst_rows)*dst_rows_max
        #dst_cols=dst_cols/np.max(dst_cols)*dst_cols_max
    elif transform =='thetascale':
        dst_rows_max=np.max(dst_rows)
        dst_cols_max=np.max(dst_cols)

        radius=(radius.astype(float))**1.25
        theta=theta*0.25
        dst_rows=radius*np.sin(theta)+centre[0]
        dst_cols=radius*np.cos(theta)+centre[1]

        #dst_rows=dst_rows/np.max(dst_rows)*dst_rows_max
        #dst_cols=dst_cols/np.max(dst_cols)*dst_cols_max



    elif transform=='sine':
        dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
        dst_cols = src[:, 0]
        dst_rows *= 1.5
        dst_rows -= 1.5 * 50


    
    dst = np.vstack([dst_cols, dst_rows]).T


    # Calculate the corners of the rotated image


    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    return tform, dst,src, rows, cols,dst_cols,dst_rows



fig,ax=plt.subplots(1,2,sharey=True,sharex=True)

load_image=2
if load_image==1:
    image = ski.data.coins()
elif load_image==2:
    im_str='www/DTArgyropelecusOlfersi.png'
    image=imread(im_str)

elif load_image==3:
    image=imread('posts/DTScarus.png')
else:
    image=imread('posts/TestPhoto.jpg')

image=image[-1:0:-1,:]

tform_method=2
if tform_method==1: # sim transform
    tform = SimilarityTransform(scale=0.5,translation=(0, 0),rotation=0.0*np.pi/4)
    new_image = warp(image, tform)
elif tform_method==2:
    
    transform_vec=['identity','yscale','yscaleDarcy299Fig151','yscaleDarcy299Fig149','yscaleDarcy299Fig147','yscalelog','yscalepower','radialscale','thetascale']

    transform=transform_vec[5]
    tform,dst,src,rows, cols,dst_cols,dst_rows= ChooseTransform(transform,image.shape)


   
    #centre=[cols/2.0,rows/2.0]
    #radius=np.sqrt((dst_rows-centre[0])**2+(dst_cols-centre[1])**2)
    #theta=np.arctan2(dst_rows-centre[0],dst_cols-centre[1])

    
    corners = np.array([[0, 0], [0, image.shape[0]], [image.shape[1], 0], [image.shape[1], image.shape[0]]])
    tformed_corners = tform(corners)

    

    minc = tformed_corners.min(axis=0)
    maxc = tformed_corners.max(axis=0)
    print(maxc)
    print(minc)
    output_shape = 2*(np.ceil(maxc)[::-1]-np.ceil(minc)[::-1]).astype(int)
    print(output_shape)
    

    tform2 = PiecewiseAffineTransform()
    dst2_rows = src[:, 1] 
    dst2_cols = src[:, 0]
    dst2_rows=dst2_rows-minc[1]
    dst2_cols=dst2_cols-minc[0]

    dst2 = np.vstack([dst2_cols, dst2_rows]).T

    tform2.estimate(src, dst2)
    tform2=AffineTransform(translation=-minc)

    #new_image = warp(image, tform2.inverse,order=0,output_shape=output_shape)

    
    # Calculate the necessary output shape
    #new_image = warp(image, (tform+tform2).inverse, output_shape=(int(image.shape[0] * 1.5), int(image.shape[1] * 1.5))).shape
    # Calculate the translation required to move all points into positive coordinates
    min_coords = dst.min(axis=0)
    translation = -min_coords
    tform_shift = AffineTransform(translation=translation)

    # Apply the warp with translation to ensure positive coordinates
    output_shape = (int(rows + 2*translation[1]), int(cols + 2*translation[0]))
    new_image_trans = warp(image, tform_shift.inverse , output_shape=output_shape)
    
    #tform,dst,src= ChooseTransform(transform)
    tform,dst,src,rows, cols,dst_cols,dst_rows= ChooseTransform(transform,new_image_trans.shape)

    
    new_image = warp(new_image_trans,tform.inverse)

    # Apply the warp with the calculated output shape
    #new_image = warp(image, tform.inverse, output_shape=output_shape)

    #
    #fig, ax = plt.subplots()
    #ax.imshow(new_image)
    #ax.axis('equal')
    #plt.show()
    #ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    #plt.show()
    #ax.axis((0, out_cols, out_rows, 0))
elif tform_method==3:
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    dst_cols = src[:, 0]
    dst_rows *= 1.5
    dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T


    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0] - 1.5 * 50
    out_cols = cols
    new_image = warp(image, tform, output_shape=(out_rows, out_cols))

    fig, ax = plt.subplots()
    ax.imshow(new_image)
    #ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    #plt.show()
    #ax.axis((0, out_cols, out_rows, 0))


# ... or any other NumPy array!
#edges = ski.filters.sobel(image)
    
x_plt=np.linspace(0,cols,cols)
y_plt=np.linspace(0,rows,rows)

[x_plt_mesh,y_plt_mesh]=np.meshgrid(x_plt,y_plt)


scale_x=cols/np.max(dst_cols)
scale_y=rows/np.max(dst_rows)

new_x_plt=np.linspace(0,cols*scale_x,cols)
new_y_plt=np.linspace(0,rows*scale_y,rows)
[newx_plt_mesh,newy_plt_mesh]=np.meshgrid(new_x_plt,new_y_plt)


#ax[0].pcolor(x_plt_mesh,y_plt_mesh,image)
#ax[1].pcolor(x_plt_mesh,y_plt_mesh,new_image)
ax[0].pcolor(image)
#ax[1].pcolor(new_image_trans)

ax[1].pcolor(new_image)

plt.show()


save_image_as_Str=True
if save_image_as_Str:
    import pickle


    # Read the encoded string from the text file
    # Read image file
    with open('data.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(image, f, pickle.HIGHEST_PROTOCOL)
    
   


    







