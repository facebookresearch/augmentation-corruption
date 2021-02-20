# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Augmentation
from scipy.ndimage import gaussian_filter
from .utils.severity import float_parameter, int_parameter, sample_level
from .utils.image import bilinear_interpolation, smoothstep
import numpy as np

class CausticRefraction(Augmentation):

    tags = ['distortion']
    name = 'caustic_refraction'

    def sample_parameters(self):
        time = np.random.uniform(low=0.5, high=2.0)
        size = np.random.uniform(low=0.75, high=1.25) * self.im_size
        #size = self.im_size
        eta = 4.0
        lens_scale = float_parameter(sample_level(self.severity, self.max_intensity), 0.5*self.im_size)
        lighting_amount = float_parameter(sample_level(self.severity, self.max_intensity), 2.0)
        softening = 1

        return { 'time' : time, 'size' : size, 'eta' : eta, 'lens_scale' : lens_scale, 'lighting_amount': lighting_amount, 'softening' : softening}

    def transform(self, image, time, size, eta, lens_scale, lighting_amount, softening):

        def caustic_noise_kernel(point, time, size):
            point = point / size
            p = (point % 1) * 6.28318530718 - 250

            i = p.copy()
            c = 1.0
            inten = 0.005

            for n in range(5):
                t = time * (1.0 - (3.5 / (n+1)))
                i = p + np.array([np.cos(t-i[0])+np.sin(t+i[1]),np.sin(t-i[1])+np.cos(t+i[0])])
                length = np.sqrt((p[0] / (np.sin(i[0]+t)/inten))**2 + (p[1] / (np.cos(i[1]+t)/inten))**2)
                c += 1.0/length

            c /= 5.0
            c = 1.17 - c ** 1.4
            color = np.clip(np.abs(c) ** 8.0, 0, 1) 
            return np.array([color, color, color])


        def refract(incident, normal, eta):
            if np.abs(np.dot(incident, normal)) >= 1.0 - 1e-3:
                return incident
            angle = np.arccos(np.dot(incident, normal))
            out_angle = np.arcsin(np.sin(angle) / eta)
            out_unrotated = np.array([np.cos(out_angle), np.sin(out_angle), 0.0])
            spectator_dim = np.cross(incident, normal)
            spectator_dim /= np.linalg.norm(spectator_dim)
            orthogonal_dim = np.cross(normal, spectator_dim)
            rotation_matrix = np.stack((normal, orthogonal_dim, spectator_dim), axis=0)
            return np.matmul(np.linalg.inv(rotation_matrix), out_unrotated)

        def luma_at_offset(image, origin, offset):
            pixel_value = image[origin[0]+offset[0], origin[1]+offset[1], :]\
                    if origin[0]+offset[0] >= 0 and origin[0]+offset[0] < image.shape[0]\
                    and origin[1]+offset[1] >= 0 and origin[1]+offset[1] < image.shape[1]\
                    else np.array([0.0,0.0,0])
            return np.dot(pixel_value, np.array([0.2126, 0.7152, 0.0722]))

        def luma_based_refract(point, image, caustics, eta, lens_scale, lighting_amount):
            north_luma = luma_at_offset(caustics, point, np.array([0,-1]))
            south_luma = luma_at_offset(caustics, point, np.array([0, 1]))
            west_luma = luma_at_offset(caustics, point, np.array([-1, 0]))
            east_luma = luma_at_offset(caustics, point, np.array([1,0]))

            lens_normal = np.array([east_luma - west_luma, south_luma - north_luma, 1.0])
            lens_normal = lens_normal / np.linalg.norm(lens_normal)

            refract_vector = refract(np.array([0.0, 0.0, 1.0]), lens_normal, eta) * lens_scale
            refract_vector = np.round(refract_vector, 3)

            #print(refract_vector)
            out_pixel = bilinear_interpolation(image, point+refract_vector[0:2])
            out_pixel += (north_luma - south_luma) * lighting_amount
            out_pixel += (east_luma - west_luma) * lighting_amount

            return np.clip(out_pixel, 0, 1)

        noise = np.array([[caustic_noise_kernel(np.array([y,x]), time, size)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = gaussian_filter(noise, sigma=softening)

        image = image.astype(np.float32) / 255
        out = np.array([[luma_based_refract(np.array([y,x]), image, noise, eta, lens_scale, lighting_amount)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip((out * 255).astype(np.uint8), 0, 255)
        

class PinchAndTwirl(Augmentation):

    tags = ['distortion']
    name = 'pinch_and_twirl'

    def sample_parameters(self):
        centers = [np.random.randint(low=0, high=self.im_size, size=2) for i in range(5)]
        radius = self.im_size // 4
        #amounts = np.random.uniform(low=0.2, high=1.0, size=5)
        #angles = np.random.uniform(low=-np.pi, high=np.pi, size=5)
        angles = [float_parameter(sample_level(self.severity, self.max_intensity), np.pi/4)-float_parameter(sample_level(self.severity, True), np.pi/8)\
                for i in range(5)]
        amounts = [float_parameter(sample_level(self.severity, self.max_intensity), 0.4) + 0.1\
                for i in range(5)]

        return {'centers' : centers, 'radius' : radius, 'amounts' : amounts, 'angles' : angles}

    def transform(self, image, centers, radius, amounts, angles):

        def warp_kernel(point, center, radius, amount, angle):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            dist = np.linalg.norm(point - center)

            if dist > radius or np.round(dist, 3) == 0.0:
                return point

            d = dist / radius
            t = np.sin(np.pi * 0.5 * d) ** (- amount)

            dx *= t
            dy *= t

            e = 1 - d
            a = angle * (e ** 2)
            
            out = center + np.array([dx*np.cos(a) - dy*np.sin(a), dx*np.sin(a) + dy*np.cos(a)])

            return out

        image = image.astype(np.float32)
        for center, angle, amount in zip(centers, angles, amounts):
            image = np.array([[bilinear_interpolation(image, warp_kernel(np.array([y,x]), center, radius, amount, angle))\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(image, 0, 255).astype(np.uint8)


class PinchAndTwirlV2(Augmentation):

    tags = ['distortion']
    name = 'pinch_and_twirl_v2'

    def sample_parameters(self):
        num_per_axis = 5 if self.im_size==224 else 3
       #angles = np.array([float_parameter(sample_level(self.severity, self.max_intensity), np.pi)-float_parameter(sample_level(self.severity, True), np.pi/2)\
       #         for i in range(num_per_axis ** 2)]).reshape(num_per_axis, num_per_axis)
        #if self.im_size == 224:
        angles = np.array([np.random.choice([1,-1]) * float_parameter(sample_level(self.severity, self.max_intensity), np.pi/2) for i in range(num_per_axis ** 2)]).reshape(num_per_axis, num_per_axis)
        #else:
        #    angles = np.array([np.random.choice([1,-1]) * (float_parameter(sample_level(self.severity, self.max_intensity), np.pi/4)+np.pi/4) for i in range(num_per_axis ** 2)]).reshape(num_per_axis, num_per_axis)

        amount = float_parameter(sample_level(self.severity, self.max_intensity), 0.4) + 0.1
        return {'num_per_axis' : num_per_axis, 'angles' : angles, 'amount' : amount}

    def transform(self, image, num_per_axis, angles, amount):

        def warp_kernel(point, center, radius, amount, angle):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            dist = np.linalg.norm(point - center)

            if dist > radius or np.round(dist, 3) == 0.0:
                return point

            d = dist / radius
            t = np.sin(np.pi * 0.5 * d) ** (- amount)

            dx *= t
            dy *= t

            e = 1 - d
            a = angle * (e ** 2)
            
            out = center + np.array([dx*np.cos(a) - dy*np.sin(a), dx*np.sin(a) + dy*np.cos(a)])

            return out

        out = image.copy().astype(np.float32)
        grid_size = self.im_size // num_per_axis
        radius = grid_size / 2
        for i in range(num_per_axis):
            for j in range(num_per_axis):
                l, r = i * grid_size, (i+1) * grid_size
                u, d = j * grid_size, (j+1) * grid_size
                center = np.array([u+radius, l+radius])
                out[u:d,l:r,:] = np.array([[bilinear_interpolation(out, warp_kernel(np.array([y,x]), center, radius, amount, angles[i,j]))\
                        for x in np.arange(l,r)] for y in np.arange(u,d)])

        return np.clip(out, 0, 255).astype(np.uint8)


class FishEye(Augmentation):

    tags = ['distortion']
    name = 'fish_eye'

    def sample_parameters(self):
        centers = [np.random.randint(low=0, high=self.im_size, size=2) for i in range(5)]
        etas = [float_parameter(sample_level(self.severity, self.max_intensity), 1.0)+1.0\
                for i in range(5)]
        radii = [np.random.uniform(low=0.1, high=0.3) * self.im_size for i in range(5)]

        

        return {'centers' : centers, 'radii' : radii, 'etas': etas}

    def transform(self, image, centers, radii, etas):
        
        def warp_kernel(point, center, a, b, eta):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            x2 = dx**2
            y2 = dy**2
            a2 = a**2
            b2 = b**2
            if (y2 >= (b2 - b2*x2/a2)):
                return point

            r = 1.0 / eta
            z = np.sqrt((1.0 - x2/a2 - y2/b2) * (a*b))
            z2 = z**2

            x_angle = np.arccos(dx / np.sqrt(x2+z2))
            angle_1 = np.pi/2 - x_angle
            angle_2 = np.arcsin(np.sin(angle_1)*r)
            angle_2 = np.pi/2 - x_angle - angle_2
            out_x = point[0] - np.tan(angle_2)*z
            #print(np.tan(angle_2)*z)

            y_angle = np.arccos(dy / np.sqrt(y2+z2))
            angle_1 = np.pi/2 - y_angle
            angle_2 = np.arcsin(np.sin(angle_1)*r)
            angle_2 = np.pi/2 - y_angle - angle_2
            out_y = point[1] - np.tan(angle_2)*z

            return np.array([out_x, out_y])

        for center, radius, eta in zip(centers, radii, etas):
            image = np.array([[bilinear_interpolation(image, warp_kernel(np.array([y,x]), center, radius, radius, eta))\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(image, 0, 255).astype(np.uint8)


class FishEyeV2(Augmentation):

    tags = ['distortion']
    name = 'fish_eye_v2'

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        #density = float_parameter(sample_level(self.severity, self.max_intensity), 0.01)
        density = 0.01 * 224**2 / (self.im_size**2)
        #eta = 2
        eta = float_parameter(sample_level(self.severity, self.max_intensity), 2.0) + 1.0
        radius = max(0.05 * self.im_size, 3)

        return {'seed' : seed, 'density' : density, 'eta': eta, 'radius' : radius}

    def transform(self, image, density, eta, radius, seed):
        
        def warp_kernel(point, center, a, b, eta):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            x2 = dx**2
            y2 = dy**2
            a2 = a**2
            b2 = b**2
            if (y2 >= (b2 - b2*x2/a2)):
                return point

            r = 1.0 / eta
            z = np.sqrt((1.0 - x2/a2 - y2/b2) * (a*b))
            z2 = z**2

            x_angle = np.arccos(dx / np.sqrt(x2+z2))
            angle_1 = np.pi/2 - x_angle
            angle_2 = np.arcsin(np.sin(angle_1)*r)
            angle_2 = np.pi/2 - x_angle - angle_2
            out_x = point[0] - np.tan(angle_2)*z
            #print(np.tan(angle_2)*z)

            y_angle = np.arccos(dy / np.sqrt(y2+z2))
            angle_1 = np.pi/2 - y_angle
            angle_2 = np.arcsin(np.sin(angle_1)*r)
            angle_2 = np.pi/2 - y_angle - angle_2
            out_y = point[1] - np.tan(angle_2)*z

            return np.array([out_x, out_y])

        random_state = np.random.RandomState(seed=seed)
        num = int(density * self.im_size**2)

        out = image.copy().astype(np.float32)
        for i in range(num):
            center = random_state.uniform(low=0, high=self.im_size, size=2)
            l = max(np.floor(center[1]-radius).astype(np.int), 0)
            r = min(np.ceil(center[1]+radius).astype(np.int), self.im_size)
            u = max(np.floor(center[0]-radius).astype(np.int), 0)
            d = min(np.ceil(center[0]+radius).astype(np.int), self.im_size)
            out[u:d,l:r,:] = np.array([[bilinear_interpolation(out, warp_kernel(np.array([y,x]), center, radius, radius, eta)) for x in np.arange(l,r)] for y in np.arange(u,d)])

        return np.clip(out, 0, 255).astype(np.uint8)


class WaterDrop(Augmentation):
    
    tags = ['distortion']
    name = 'water_drop'

    def sample_parameters(self):
        center = np.array([self.im_size //2, self.im_size//2])
        center = np.random.uniform(low=0.25, high=0.75, size=2) * self.im_size
        radius = self.im_size//2
        amplitude = float_parameter(sample_level(self.severity, self.max_intensity), 0.25)
        wavelength = np.random.uniform(low=0.05, high=0.2) * self.im_size
        phase = np.random.uniform(low=0.0, high=2*np.pi)

        return {'center': center, 'radius' : radius, 'amplitude' : amplitude, 'wavelength' : wavelength, 'phase': phase}

    def transform(self, image, center, radius, amplitude, wavelength, phase):

        def warp_kernel(point, center, radius, amplitude, wavelength, phase):

            dx, dy = point - center
            dist = np.linalg.norm(point-center)
            if dist > radius:
                return point

            amount = amplitude * np.sin(dist / wavelength * np.pi * 2 - phase)
            if dist != 0.0:
                amount *= wavelength / dist
            return point + amount * (point - center)


        image = np.array([[bilinear_interpolation(image, warp_kernel(np.array([y,x]), center, radius, amplitude, wavelength, phase))\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8)

class Ripple(Augmentation):

    tags = ['distortion']
    name = 'ripple'

    def sample_parameters(self):
        amplitudes = np.array([float_parameter(sample_level(self.severity, self.max_intensity), 0.025)\
                for i in range(2)]) * self.im_size
        wavelengths = np.random.uniform(low=0.1, high=0.3, size=2) * self.im_size
        phases = np.random.uniform(low=0, high=2*np.pi, size=2)
        return {'amplitudes' : amplitudes, 'wavelengths' : wavelengths, 'phases' : phases}

    def transform(self, image, wavelengths, phases, amplitudes):

        def warp_kernel(point, wavelengths, phases, amplitudes):
            return point + amplitudes * np.sin(2 * np.pi * point / wavelengths + phases)

        image = np.array([[bilinear_interpolation(image, warp_kernel(np.array([y,x]), wavelengths, phases, amplitudes))\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8) 

class ColorHalfTone(Augmentation):

    tags = ['distortion']
    name = 'color_half_tone'

    def sample_parameters(self):
        #angles = np.array([108, 162, 90]) * np.pi/180
        angles = np.random.uniform(low=0, high=2*np.pi, size=3)
        dot_area = float_parameter(sample_level(self.severity, self.max_intensity), 9*np.pi)
        dot_radius = np.sqrt(dot_area/np.pi)

        return {'angles' : angles, 'dot_radius' : dot_radius}

    def transform(self, image, angles, dot_radius):

        grid_size = 2 * dot_radius * 1.414
        mx = [0, -1, 1, 0, 0]
        my = [0, 0, 0, -1, 1]
        out = np.zeros_like(image)
        for y in range(self.im_size):
            for c in range(3):
                angle = angles[c]
                cos = np.cos(angle)
                sin = np.sin(angle)
                for x in range(self.im_size):
                    tx = cos * x + sin * y
                    ty = - sin * x + cos * y

                    tx = tx - (tx - grid_size/2) % grid_size + grid_size/2
                    ty = ty - (ty - grid_size/2) % grid_size + grid_size/2

                    f = 1
                    for i in range(5):
                        ttx = tx + mx[i]*grid_size
                        tty = ty + my[i]*grid_size

                        ntx = cos * ttx - sin * tty
                        nty = sin * ttx + cos * tty

                        nx = np.clip(int(ntx), 0, self.im_size-1)
                        ny = np.clip(int(nty), 0, self.im_size-1)

                        l = image[nx, ny, c] / 255
                        l = 1 - l**2
                        l *= grid_size/2 * 1.414
                        dx = x-ntx
                        dy = y-nty
                        r = np.linalg.norm(np.array([dx,dy]))
                        f2 = 1-smoothstep(r, r+1, l)
                        f = min(f, f2)

                    out[x, y, c] = f

        return np.clip(255 * out, 0, 255).astype(np.uint8)        
