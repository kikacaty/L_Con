# -*- coding: utf-8 -*-

# set the colormap and centre the colorbar
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

#import imageio
#from moviepy.editor import ImageSequenceClip
import os

# %matplotlib inline

class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

    
import re
# import numpy as np
#import pyprind
# parsing intermediate data -> roi_pcls, features
def parse_log(log_file_name, n_frames):
    # pcl
    roi_pcl_frames = []

    # features
    max_height_frames = []
    mean_height_frames = []
    count_frames = []
    direction_frames = []
    top_intensity_frames = []
    mean_intensity_frames = []
    distance_frames = []
    nonempty_frames = []
    objs_frames = []
    bar = pyprind.ProgBar(n_frames, bar_char='█')
    with open(log_file_name) as f:
        while True:
            line = f.readline()
            parsed_line = line[line.find("##Data##")+len("##Data##"):]
            if parsed_line.find("ROI_CLOUD Start") != -1:
                # collecting roi pcl
                roi_pcl = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("ROI_CLOUD End") == -1:
                    if parsed_line.find("TS") != -1:
                        ts = parsed_line[parsed_line.find("TS")+len("TS: "):]
                    else:
                        #store point in roi pcl
                        p_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                        p = [float(num) for num in p_str]
                        roi_pcl.append(p)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # roi pcl frame end
                roi_pcl_frames.append(roi_pcl)
                bar.update(len(roi_pcl_frames))
                if len(roi_pcl_frames) > n_frames:
                    break

            elif parsed_line.find("MAX Height Start") != -1:
                # collecting features
                feature = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("MAX Height End") == -1:
                    #store features
                    v_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                    v = float(v_str[0])
                    feature.append(v)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # feature frame end
                max_height_frames.append(feature)

            elif parsed_line.find("MEAN Height Start") != -1:
                # collecting features
                feature = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("MEAN Height End") == -1:
                    #store features
                    v_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                    v = float(v_str[0])
                    feature.append(v)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # feature frame end
                mean_height_frames.append(feature)

            elif parsed_line.find("Count Start") != -1:
                # collecting features
                feature = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("Count End") == -1:
                    #store features
                    v_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                    v = float(v_str[0])
                    feature.append(v)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # feature frame end
                count_frames.append(feature)

            elif parsed_line.find("Direction Start") != -1:
                # collecting features
                feature = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("Direction End") == -1:
                    #store features
                    v_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                    v = float(v_str[0])
                    feature.append(v)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # feature frame end
                direction_frames.append(feature)

            elif parsed_line.find("Top Intensity Start") != -1:
                # collecting features
                feature = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("Top Intensity End") == -1:
                    #store features
                    v_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                    v = float(v_str[0])
                    feature.append(v)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # feature frame end
                top_intensity_frames.append(feature)

            elif parsed_line.find("Mean Intensity Start") != -1:
                # collecting features
                feature = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("Mean Intensity End") == -1:
                    #store features
                    v_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                    v = float(v_str[0])
                    feature.append(v)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # feature frame end
                mean_intensity_frames.append(feature)

            elif parsed_line.find("Dist Start") != -1:
                # collecting features
                feature = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("Dist End") == -1:
                    #store features
                    v_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                    v = float(v_str[0])
                    feature.append(v)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # feature frame end
                distance_frames.append(feature)

            elif parsed_line.find("Nonempt Start") != -1:
                # collecting features
                feature = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("Nonempt End") == -1:
                    #store features
                    v_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                    v = float(v_str[0])
                    feature.append(v)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # feature frame end
                nonempty_frames.append(feature)
            elif parsed_line.find("Features") != -1:
                continue
            elif parsed_line.find("Builder Start") != -1:
                # collecting objects
                objs = []
                line = f.readline()
                parsed_line = line[line.find("##Data##")+len("##Data##"):]
                while parsed_line.find("Builder End") == -1:
                    #store features
                    v_str = re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", parsed_line)
                    v = [float(item) for item in v_str]
                    objs.append(v)
                    line = f.readline()
                    parsed_line = line[line.find("##Data##")+len("##Data##"):]

                # feature frame end
                objs_frames.append(objs)
            else:
                print parsed_line
                continue
    return roi_pcl_frames, [max_height_frames, mean_height_frames, count_frames, direction_frames, top_intensity_frames, mean_intensity_frames, distance_frames, nonempty_frames], objs_frames

def draw_feature(feature):
    if feature.shape != (512,512):
        feature = np.reshape(feature,(512,512))
    import matplotlib.pylab as plt
    from tools import MidpointNormalize
#     %matplotlib inline
    plt.figure(figsize=(10,10))
    plt.imshow(feature, cmap='RdBu_r', interpolation='nearest', norm = MidpointNormalize(midpoint=0))
    plt.colorbar()
    plt.show()
    
def draw_feature_frame(feature, idx, name):
    if feature.shape != (512,512):
        feature = np.reshape(feature,(512,512))
        
    filename = 'video/{1}_frame_{0:0>4}.png'.format(idx,name)
    
#     %matplotlib inline
    import matplotlib.pylab as plt
    from tools import MidpointNormalize
    plt.figure(figsize=(10,10))
    plt.imshow(feature, cmap='RdBu_r', interpolation='nearest', norm = MidpointNormalize(midpoint=0))
    plt.colorbar()
    plt.title("{1} Frame: {0:0>4}".format(idx,name))
    plt.savefig(filename)
    plt.close()
    return filename

def draw_annotated_feature(feature,objs):
    import math
    if feature.shape != (512,512):
        feature = np.reshape(feature,(512,512))
#     %matplotlib inline
    import matplotlib.pylab as plt
    from tools import MidpointNormalize
    f,ax = plt.subplots(1,1,figsize=(10,10))
    plt.imshow(feature, cmap='RdBu_r', interpolation='nearest', norm = MidpointNormalize(midpoint=0))
    for obj in objs:
        path = [(PCL2GridP(p[0]),PCL2GridP(p[1])) for p in obj]
        poly = matplotlib.patches.Polygon(path,fill = False,color='g')
        ax.add_patch(poly)
    plt.colorbar()
    plt.show()
def PCL2GridP(x):
    grid_size = 512
    scale = 0.5 * grid_size/60.0
    return x * scale + grid_size * 0.5

def PCL2GridDis(x):
    grid_size = 512
    scale = grid_size/60.0
    return x * scale

def draw_pcl(pcl):
#     %matplotlib inline
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    # import seaborn as sns; sns.set()
    
    roi_pcl = np.array(pcl)
    px = [p[0] for p in roi_pcl]
    py = [p[1] for p in roi_pcl]
    pz = [p[2] for p in roi_pcl]
    pi = [p[4] for p in roi_pcl]
    n = len(px)
    point_size = 0.05
    axes_limits = [
        [-80,80], # X axis
        [-15,15], # Y axis
        [-3,3] # Z axis
    ]
    
    # Plot x-y, y-z, x-z
    f,ax = plt.subplots(3,1,figsize=(20,5*3))
    # ax = fig.add_subplot(111, projection='3d')
    ax[0].scatter(px,py,c = pi,cmap='gray',s=point_size)
    ax[0].set_xlabel('X axis')
    ax[0].set_ylabel('Y axis')
    ax[0].set_xlim(axes_limits[0])
    ax[0].set_ylim(axes_limits[1])
    ax[1].scatter(py,pz,c = pi,cmap='gray',s=point_size)
    ax[1].set_xlabel('Y axis')
    ax[1].set_ylabel('Z axis')
    ax[1].set_xlim(axes_limits[1])
    ax[1].set_ylim(axes_limits[2])
    ax[2].scatter(px,pz,c = pi,cmap='gray',s=point_size)
    ax[2].set_xlabel('X axis')
    ax[2].set_ylabel('Z axis')
    ax[2].set_xlim(axes_limits[0])
    ax[2].set_ylim(axes_limits[2])
    plt.show()

def draw_pcl_frames(pcl_frames, idx):
    
    filename = 'video/frame_{0:0>4}.png'.format(idx)
    
    # if exist, skip
#     import os  
#     if os.path.isfile(filename): 
#         return filename
        
#     %matplotlib inline
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    # import seaborn as sns; sns.set()
    
    roi_pcl = np.array(pcl_frames[idx])
    px = [p[0] for p in roi_pcl]
    py = [p[1] for p in roi_pcl]
    pz = [p[2] for p in roi_pcl]
    pi = [p[4] for p in roi_pcl]
    n = len(px)
    point_size = 0.05
    axes_limits = [
        [-80,80], # X axis
        [-15,15], # Y axis
        [-3,3] # Z axis
    ]
    
    # Plot x-y, y-z, x-z
    f,ax = plt.subplots(3,1,figsize=(20,5*3))
    # ax = fig.add_subplot(111, projection='3d')
    ax[0].scatter(px,py,c = pi,cmap='gray',s=point_size)
    ax[0].set_xlabel('X axis')
    ax[0].set_ylabel('Y axis')
    ax[0].set_xlim(axes_limits[0])
    ax[0].set_ylim(axes_limits[1])
    ax[1].scatter(py,pz,c = pi,cmap='gray',s=point_size)
    ax[1].set_xlabel('Y axis')
    ax[1].set_ylabel('Z axis')
    ax[1].set_xlim(axes_limits[1])
    ax[1].set_ylim(axes_limits[2])
    ax[2].scatter(px,pz,c = pi,cmap='gray',s=point_size)
    ax[2].set_xlabel('X axis')
    ax[2].set_ylabel('Z axis')
    ax[2].set_xlim(axes_limits[0])
    ax[2].set_ylim(axes_limits[2])
    
    for sub_ax in ax:
        sub_ax.set_title("Frame: {0:0>4}".format(idx))
    
    plt.savefig(filename)
    plt.close(f)
    return filename

def pcl_demo(roi_pcl_frames, name = "pcl", fps = 1, n_frames=0):
    if n_frames == 0:
        n_frames = len(roi_pcl_frames)
    frames = []
    if n_frames > len(roi_pcl_frames):
        n_frames = len(roi_pcl_frames)

    bar = pyprind.ProgBar(n_frames, bar_char='█')

    print('Preparing animation frames...')
    for i in range(n_frames):
        bar.update(i)
        filename = draw_pcl_frames(roi_pcl_frames, i)
        frames += [filename]
    print('...Animation frames ready.')

    gif_filename = '{0}_data.gif'.format(name)

    clip = ImageSequenceClip(frames, fps=fps)
    try:
        os.remove(gif_filename)
    except OSError:
        pass
    clip.write_gif(gif_filename, fps=fps)
    
def feature_demo(feature_frames, name, fps = 1):
    frames = []
    n_frames = len(feature_frames)

    bar = pyprind.ProgBar(n_frames, bar_char='█')

    print('Preparing animation frames...')
    for i in range(n_frames):
        bar.update(i)
        filename = draw_feature_frame(np.array(feature_frames[i]), i, name)
        frames += [filename]
    print('...Animation frames ready.')

    gif_filename = '{0}_data.gif'.format(name)

    clip = ImageSequenceClip(frames, fps=fps)
    try:
        os.remove(gif_filename)
    except OSError:
        pass
    clip.write_gif(gif_filename, fps=fps)
    
    
# IO
# dump ml outputs
def dump_cnn_out(output_data,filename = 'cnn.out'):
    with open('/data/yulongc/apollo/data/percpt/'+filename,'w') as f:
        # category
        for p in output_data[0][0].flatten():
            f.write('%.15f '%(p))
        # class
        f.write('\n')
        for p_class in output_data[1]:
            for p in p_class.flatten():
                f.write('%.15f '%(p))
            f.write('\n')
        # confidence
        for p in output_data[2][0].flatten():
            f.write('%.15f '%(p))
        f.write('\n')
    #     # heading
    #     for heading in output_data[3]:
    #         for p in heading:
    #             f.write('%.15f '%(p))
        # height
        for p in output_data[3][0].flatten():
            f.write('%.15f '%(p))
        f.write('\n')
        # instance
        for instance in output_data[4]:
            for p in instance.flatten():
                f.write('%.15f '%(p))
            f.write('\n')

# read obstacles
def read_objs():
    obs_polygon = []
    with open('/data/yulongc/apollo/data/percpt/builder.out','rb') as f:
        while True:
            line = f.readline().split(' ')
            if line[0] == '':
                break
            poly_size = int(line[8])
            poly = []
            for i in range(poly_size):
                poly.append((float(line[9+i*3]),float(line[10+i*3]),float(line[11+i*3])))
            obs_polygon.append(poly)

    return obs_polygon