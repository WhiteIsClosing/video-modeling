import numpy
from scipy import misc

def tuple_add(a, b):
    """
    Add two tuples with arbitrary size. 
    """
    assert len(a) == len(b)
    return tuple(x + y for x, y in zip(a, b))


# translation 
def translate_gen(imgin, num_seq, seq_len, patch_size, 
                    maxv0=[3, 3],# maxa0=[1, 1],
                    en_acc=False,
                    shift_margin=[0, 0]):
    """
    Generate translation patches from imgin(non-flattened).  

    Parameters
    ----------
    imgin: double
        The non-flattened single-channel input image.
    num_seq: int
        Number of sequences to generate. 
    seq_len: int
        The length of the output patch sequence.
    patch_size: int
        The length of side of the output pathes.
    maxv0: double (1, 2)
        Maximal initial absolute value of velocity along x and y axis. 
    # maxa0: double (1, 2)
    #     Maximal initial absolute value of acceleration along x and y axis. 
    en_acc: bool
        If add acceleration into the translation.
    shift_margin: int (1, 2)
        Control the maximal shifting distance.

    Returns
    -------
    patch_seq: double (num_seq, seq_len * patch_size**2)
        Output patch sequences.
    """

    img_shape = imgin.shape
    
    offsets = [None] * seq_len    # index: (row, column)
    seqs = numpy.zeros((num_seq, seq_len*(patch_size**2)))


    for i in range(num_seq):

        # generate shift sequence
        valid_shift = False

        # generate offset sequences until it is valid
        while not valid_shift:
            valid_vel = True

            s = (0, 0)  # coordinates: (x, y)
            v = (numpy.random.randint(-maxv0[0], maxv0[0]),
                numpy.random.randint(-maxv0[1], maxv0[1]))
            if en_acc:
                a = (numpy.random.randn()/2, numpy.random.randn()/2)
            else:
                a = (0., 0.)

            if int(v[0]) == 0 and (v[1]) == 0:
                # valid_vel = False   # initial translation can't be too slow
                continue

            # scope of translation offset
            shift_top = 0
            shift_bottom = 0
            shift_left = 0
            shift_right = 0

            # generate an offset sequence
            for t in range(seq_len):
                s_idx = [int(s[1]), int(s[0])]
                offsets[t] = s_idx

                shift_top = min(s_idx[0], shift_top)
                shift_bottom = max(s_idx[0], shift_bottom)
                shift_left = min(s_idx[1], shift_left)
                shift_right = max(s_idx[1], shift_right)

                if v[0] >= patch_size/4 or v[1] >= patch_size/4:
                    valid_vel = False   # translation can't be too fast

                s = tuple_add(s, v)
                v = tuple_add(v, a)

            shift_region = [shift_bottom - shift_top, shift_right - shift_left]
            if shift_region[0] < img_shape[0]-patch_size-shift_margin[0] and\
                shift_region[1] < img_shape[1]-patch_size-shift_margin[1]:
                valid_shift = True and valid_vel

        # valid sampling regions on the image matrix
        valid_left = 0-shift_left
        valid_right = img_shape[1]-shift_right-patch_size
        valid_top = 0-shift_top
        valid_bottom = img_shape[0]-shift_bottom-patch_size
        
        # sample an initial position
        row0 = numpy.random.randint(high=valid_bottom, low=valid_top)  
        col0 = numpy.random.randint(high=valid_right, low=valid_left)  

        # copy patches from the image
        seqs[i, :] = numpy.concatenate([imgin[row0+of[0]:row0+of[0]+patch_size,
            col0+of[1]:col0+of[1]+patch_size].flatten() for of in offsets],
            axis=1)

        # try:
        #seqs[i, :] = numpy.concatenate([imgin[row0+of[0]:row0+of[0]+patch_size,
        #       col0+of[1]:col0+of[1]+patch_size].flatten() for of in offsets],
        #         axis=1)
        # except:
        #     print '+++'
        #     print offsets
        #     print (shift_top, shift_bottom, shift_left, shift_right)
        #     print (valid_top, valid_bottom, valid_left, valid_right)
        #     print (row0, col0)
        #     print [(row0+of[0]+patch_size, col0+of[1]+patch_size)\
        #             for of in offsets]

    return seqs


# rotation
def rotate_gen(imgin, num_seq, seq_len, patch_size,
                rgw0=30., rga0=2.):
    """
    Generate rotation patches from imgin(non-flattened).  

    Parameters
    ----------
    imgin: double
        The non-flattened single-channel input image.
    num_seq: int
        Number of sequences to generate. 
    seq_len: int
        The length of the output patch sequence.
    patch_size: int
        The length of side of the output pathes.
    rgw0: double
        Range of initial values of angular velocity.
    rga0: double
        Range of initial values of angular acceleration.

    Returns
    -------
    patch_seq: double (num_seq, seq_len * patch_size**2)
        Output patch sequences.
    """

    img_shape = imgin.shape
    crops = patch_size/2            # crop after rotation
    crope = patch_size/2+patch_size # crop after rotation
    
    thetas = [None] * seq_len  
    seqs = numpy.zeros((num_seq, seq_len*(patch_size**2)))

    for i in range(num_seq):

        # compute the rotation angle for each time step
        theta = 0.
        w = numpy.random.uniform(-rgw0, rgw0)
        a = numpy.random.uniform(-rga0, rga0)
        for t in range(seq_len):
            thetas[t] = theta
            theta += w
            w += a

        # sample a 2 times larger patch for rotation purpose
        row = numpy.random.randint(high=img_shape[0]-2*patch_size, 
                                    low=0)  
        col = numpy.random.randint(high=img_shape[1]-2*patch_size, 
                                    low=0)  
        patch_large = imgin[row:row+2*patch_size, col:col+2*patch_size]

        seqs[i, :] = numpy.concatenate([misc.imrotate(patch_large, theta)\
                    [crops:crope, crops:crope].flatten() 
                    for theta in thetas], axis=1)
    return seqs


# scaling
def scale_gen(imgin, num_seq, seq_len, patch_size,
                rgk0=[.71, 1.4], rga0=[-.01, .01]
                ):
    """
    Generate scaling patches from imgin(non-flattened).  

    Parameters
    ----------
    imgin: double
        The non-flattened single-channel input image.
    num_seq: int
        Number of sequences to generate. 
    seq_len: int
        The length of the output patch sequence.
    patch_size: int
        The length of side of the output pathes.
    rgk0: double
        Range of the initial scaling factor.  
    # rga0: double
    #     Range of the acceleration of the scaling fatctor. 

    Returns
    -------
    patch_seq: double (num_seq, seq_len * patch_size**2)
        Output patch sequences.
    """

    img_shape = imgin.shape
    
    scales = [None] * seq_len    
    images = [None] * seq_len
    seqs = numpy.zeros((num_seq, seq_len*(patch_size**2)))

    for i in range(num_seq):

        valid_scale = False
        while not valid_scale:
            # compute the scaling factor for each time step
            k = numpy.random.uniform(rgk0[0], rgk0[1])
            a = numpy.random.uniform(rga0[0], rga0[1])

            scale = 1
            minscale = float('infinity')

            for t in range(seq_len):
                scales[t] = scale
                if scale < minscale:
                    minscale = scale
                scale *= k
                k += a

            sample_size = int(patch_size / minscale)

            if sample_size <= min(img_shape[0], img_shape[1]):
                valid_scale = True

        row = numpy.random.randint(high=img_shape[0]-sample_size, low=0)  
        col = numpy.random.randint(high=img_shape[1]-sample_size, low=0)  
        patch_sample = imgin[row:row+sample_size, col:col+sample_size]
        # print scales

        for t in range(seq_len):
            scale = scales[t] / minscale
            size = int(patch_size * scale)
            patch_resize = misc.imresize(patch_sample, (size, size))
            images[t] = patch_resize[(size-patch_size)/2:(size+patch_size)/2, 
                                        (size-patch_size)/2:(size+patch_size)/2]
            # try:
            # except:
            #     print '+++'
            #     print size
            #     print scale
            #     print k
            #     print scales
            #     print minscale

        seqs[i, :] = numpy.concatenate([image.flatten() for image in images], 
                                        axis=1)

    return seqs
