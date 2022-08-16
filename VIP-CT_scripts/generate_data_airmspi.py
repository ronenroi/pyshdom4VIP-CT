import cv2
import pandas as pd
import scipy.io as sio
import shdom
import glob
import numpy as np
import matplotlib.pyplot as plt
from shdom import CloudCT_setup
import os 
import copy
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
from scipy import ndimage
from shdom.CloudCT_Utils import *
import dill as pickle
# from shdom.rays_in_voxels import *


from shdom.AirMSPI import AirMSPIMeasurements, LinePlaneCollision

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
VISUALIZATION = False

n_jobs = 35
SCALE_LWC_BY_01 = True
IF_PAD_SIDES = True
if IF_PAD_SIDES:
    PAD_SIDES = 2
else:
    PAD_SIDES = 0
    
base_path = '/media/roironen/8AAE21F5AE21DB09/Data'
Clouds_PATH = '/media/roironen/8AAE21F5AE21DB09/Data/clouds/'
data_dir = '/media/roironen/8AAE21F5AE21DB09/Data/airmspi_projections/train/'

wavelength = 0.660

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

CLOUDS = glob.glob(Clouds_PATH + "/cloud*.txt")

for file_index, file_name in enumerate(CLOUDS):
    
    cloud_index = file_name.split("/")[-1].split(".txt")[0].split("d")[-1]
    images_list = []
    format_ = '*'  # load
    paths = sorted(glob.glob(data_dir + '/' + format_))
    n_files = len(paths)

            
    # ---------------------------------------
    # open cloud file:
    CloudFieldFile = os.path.join(Clouds_PATH, f'cloud{cloud_index}.txt')
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ----------------------------------------------------------

    mie = shdom.MiePolydisperse()
    options = {}
    options['start_reff'] = 1  # Starting effective radius [Micron]
    options['end_reff'] = 35.0
    options['num_reff'] = 100
    options['start_veff'] = 0.01
    options['end_veff'] = 0.4
    options['num_veff'] = 117
    options['radius_cutoff'] = 65.0
    options['wavelength_resolution'] = 0.001

    _ = CALC_MIE_TABLES(where_to_check_path="../mie_tables",
                        wavelength_micron=0.660,
                        options=options,
                        wavelength_averaging=False)
    mie.read_table(file_path="../mie_tables/polydisperse/Water_660nm.scat")
    surface_albedo = 0.05

    try:
        droplets = shdom.MicrophysicalScatterer()
        droplets.load_from_csv(CloudFieldFile) # I aslos used with ,veff=0.1

        lwc = droplets.lwc.data
        veff = droplets.veff.data
        veff[veff<0.02] = 0.02
        veff[veff>=0.55] = 0.55

        # used when I wanted to refine lwc scale:
        if SCALE_LWC_BY_01:
            reff = droplets.reff.data
            new_lwc = shdom.GridData(droplets.grid,lwc/10)
            new_veff = shdom.GridData(droplets.grid,veff)
            new_reff = shdom.GridData(droplets.grid,reff)
            new_droplets = shdom.MicrophysicalScatterer(new_lwc,new_reff,new_veff)
            droplets = new_droplets

        droplets.add_mie(mie)

        # Rayleigh scattering for air molecules up to 20 km
        df = pd.read_csv("../ancillary_data/AFGL_summer_mid_lat.txt", comment='#', sep=' ')
        altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
        temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
        temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
        air_grid = shdom.Grid(z=np.linspace(0, 20, 20))
        rayleigh = shdom.Rayleigh(wavelength=wavelength)
        rayleigh.set_profile(temperature_profile.resample(air_grid))
        air = rayleigh.get_scatterer()

        atmospheric_grid = droplets.grid + air.grid
        atmosphere = shdom.Medium(atmospheric_grid)

        atmosphere.add_scatterer(air, name='air')
        atmosphere.add_scatterer(droplets, name='cloud')

        # EXTRACT from GT before the padding:
        original_droplets = copy.deepcopy(droplets)
        ext = original_droplets.get_extinction(wavelength).data
        droplets_grid = original_droplets.grid
        grid = np.array([droplets_grid.x, droplets_grid.y, droplets_grid.z])


        #------------------------------------------------------
        # Pad atmospere:
        #fig1 = mlab.figure(size=(600, 600))#
        #atmosphere.show_scatterer(name='cloud')

        if PAD_SIDES is not 0:
            values = np.array(PAD_SIDES*[0.0])
            atmosphere.pad_scatterer(name = 'cloud', axis=0, right = True, values=values)
            atmosphere.pad_scatterer(name = 'cloud', axis=0, right = False, values=values)
            atmosphere.pad_scatterer(name = 'cloud', axis=1, right = True, values=values)
            atmosphere.pad_scatterer(name = 'cloud', axis=1, right = False, values=values)
            values = np.array(2*[0.0])
            atmosphere.pad_scatterer(name = 'cloud', axis=2, right = True, values=values)

        # ----------------------------------------------------------
        #center of domain
        mean_x = droplets_grid.dx*0.5*(droplets_grid.nx + 2*PAD_SIDES)
        mean_y = droplets_grid.dy*0.5*(droplets_grid.ny + 2*PAD_SIDES)
        # ----------------------------------------------------------

        numerical_params = shdom.NumericalParameters(num_mu_bins=8, num_phi_bins=16, adapt_grid_factor=5,
                                                         split_accuracy=0.1,
                                                         max_total_mb=300000.0, num_sh_term_factor=5)
        scene_params = shdom.SceneParameters(wavelength=mie.wavelength,
                                                 surface=shdom.LambertianSurface(albedo=surface_albedo),
                                                 source=shdom.SolarSource(azimuth=1.685, zenith=132.225))

        rte_solver = shdom.RteSolver(scene_params, numerical_params)
        rte_solver.set_medium(atmosphere)

        #rte_solver.init_solution()
        rte_solver.solve(maxiter=150)

        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # ----------------------------------------------------------

        # ---------------------------------------------------
        # ---------------------------------------------------
        for path in paths:
            path_stamp = path.split('/')[-1]
            Output_path = os.path.join(base_path, 'SIMULATED_AIRMSPI_TRAIN_' + path_stamp)
            if not os.path.exists(Output_path):
                os.mkdir(Output_path)
            with open(path, 'rb') as f:
                x = pickle.load(f)
            projections = x['projections']
            masks = x['masks']
        
            roi_projections = shdom.MultiViewProjection()
            for mask, projection in zip(masks,projections.projection_list):
                center_x, center_y = ndimage.center_of_mass(mask)
        
                height, width = mask.shape[:2]
                t_height = np.int(height/2 - center_x)
                t_width = np.int(width/2 - center_y)
                T = np.float32([[1, 0, t_width], [0, 1, t_height]])
        
                x_s = int(height/2) - 175
                x_e = int(height/2) + 175
                y_s = int(width/2) - 175
                y_e = int(width/2) + 175
        
        
                # assert np.all(mask[x_s:x_e,y_s:y_e])
                x = np.full(mask.shape,np.nan)
                x[mask] = projection.x
                x = cv2.warpAffine(x, T, (width, height), borderValue=np.nan)
        
        
                x = x[x_s:x_e,y_s:y_e] + mean_x
                y = np.full(mask.shape,np.nan)
                y[mask] = projection.y
                y = cv2.warpAffine(y, T, (width, height), borderValue=np.nan)
                y = y[x_s:x_e,y_s:y_e] + mean_y
        
                z = np.full(mask.shape,np.nan)
                z[mask] = projection.z
                z = cv2.warpAffine(z, T, (width, height), borderValue=np.nan)
                z = z[x_s:x_e,y_s:y_e]
        
                mu = np.full(mask.shape,np.nan)
                mu[mask] = projection.mu
                mu = cv2.warpAffine(mu, T, (width, height), borderValue=np.nan)
                mu = mu[x_s:x_e,y_s:y_e]
        
                phi = np.full(mask.shape,np.nan)
                phi[mask] = projection.phi
                phi = cv2.warpAffine(phi, T, (width, height), borderValue=np.nan)
                phi = phi[x_s:x_e,y_s:y_e]
        
                resolution = x.shape
                roi_projections.add_projection(shdom.Projection(x=x.ravel('F'), y=y.ravel('F'), z=z.ravel('F'), mu=mu.ravel('F'),
                                            phi=phi.ravel('F'),
                                            resolution=resolution))
                
            camera = shdom.Camera(shdom.RadianceSensor(), roi_projections)



            images = camera.render(rte_solver,n_jobs=40)

            new_images = []
            # get rid of nstokes dim
            for frame_index, image in enumerate(images):
                new_image = np.squeeze(image)
                new_images.append(new_image)
            images = new_images

            images_list.append(images.copy())
            print(f"cloud {cloud_index} is finished for projections of {path_stamp}.")
            # Save images:
            # save images as mat for cloudCT neural network
            filename = os.path.join(Output_path,
                                    f'satellites_images_{cloud_index}.pkl')

            print(f'saving cloud in {filename}')
            # SAVE AS ROI:
            cloud = {'images': np.array(images),
                     'cloud_path': CloudFieldFile}


            with open(filename, 'wb') as outfile:
                pickle.dump(cloud, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            # clouds loop
        print(f"------ cloud {cloud_index} is finished for all.-----")
            
    except Exception as e:
        print(f"cloud {cloud_index} failed : {e}")
    

        
print('done')
