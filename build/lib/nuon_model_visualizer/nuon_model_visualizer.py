#!/usr/bin/env python
"""Nuon model visualizer"""

# -*- coding: utf-8 -*-

#
# Copyright (c) 2023-2024, Cyrille Favreau (cyrille.favreau@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import ROOT
import numpy as np
import math
from bioexplorer import BioExplorer, TransferFunction, Vector3
import matplotlib.pyplot as plt

class NuonModelVisualizer:

    def __init__(self, bio_explorer, root_proton_file, root_collision_file, radius_multiplier=0.01, nb_sub_particles=66):
        self._bio_explorer = bio_explorer
        self._core = bio_explorer.core_api()
        self._root_proton_file = ROOT.TFile.Open(root_proton_file)
        self._root_collision_file = ROOT.TFile.Open(root_collision_file)

        self._T_protons = self._root_proton_file.Get('T')
        self._T_collisions = self._root_collision_file.Get('T')

        hinfo = self._root_proton_file.Get('hinfo')
        self._Rfactor = hinfo.GetBinContent(31)
        if self._Rfactor <= 0:
            self._Rfactor = 1.94

        self._radius_multiplier = radius_multiplier
        self._nb_sub_particles = nb_sub_particles

    def _rotate_z(self, point, angle):
        """
        Rotate a 3D point around the z-axis by a given angle.
        
        Parameters:
            point (numpy.array): The 3D point as a numpy array of shape (3,).
            angle (float): The angle of rotation in radians.
            
        Returns:
            numpy.array: The rotated 3D point.
        """
        x, y, z = point
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                    [sin_theta, cos_theta, 0],
                                    [0, 0, 1]])
        rotated_point = np.dot(rotation_matrix, point)
        return rotated_point
    
    def _rotate_vector(vector, theta, phi, psi):
        Rx = np.array([[1, 0, 0],
                    [0, math.cos(theta), -math.sin(theta)],
                    [0, math.sin(theta), math.cos(theta)]])
        
        Ry = np.array([[math.cos(phi), 0, math.sin(phi)],
                    [0, 1, 0],
                    [-math.sin(phi), 0, math.cos(phi)]])
        
        Rz = np.array([[math.cos(psi), -math.sin(psi), 0],
                    [math.sin(psi), math.cos(psi), 0],
                    [0, 0, 1]])
        rotation_matrix = Rz.dot(Ry).dot(Rx)
        return np.dot(rotation_matrix, vector)

    def _load_proton_1(self, ax, sub_particles_to_ignore, marker_size=1.0, collision_maker_size=2.0, j_cut=1.0, output_folder=None, position=[0, 0, 0], z_rotation_angle=0.0):
        i1 = getattr(self._T_collisions, 'i1')
        phi1 = getattr(self._T_collisions, 'phi1')
        
        self._T_protons.GetEntry(i1)
        vxpart1 = np.array(getattr(self._T_protons, 'x'))
        if max(vxpart1)>1e6:
            print('ERROR: ' + str(vxpart1))
            return
        vypart1 = np.array(getattr(self._T_protons, 'y'))
        vzpart1 = np.array(getattr(self._T_protons, 'z'))
        drp1 = np.array(getattr(self._T_protons, 'drp'))
        
        v1p_positions = list()
        v1p_radii = list()
        v1n_positions = list()
        v1n_radii = list()
        for i in range(self._nb_sub_particles):
            if i in sub_particles_to_ignore:
                continue
            vx1 = self._Rfactor * vxpart1[i] * math.cos(phi1) - self._Rfactor * vypart1[i] * math.sin(phi1);
            vy1 = self._Rfactor * vxpart1[i] * math.sin(phi1) + self._Rfactor * vypart1[i] * math.cos(phi1);
        
            vx1p = vx1 * (1 + drp1)
            vx1n = vx1 * (1 - drp1)
            
            vy1p = vy1 * (1 + drp1)
            vy1n = vy1 * (1 - drp1)

            z = vzpart1[i]

            point = np.array([vx1p, vy1p, z])
            rotated_v1p = self._rotate_z(point, z_rotation_angle)

            point = np.array([vx1n, vy1n, z])
            rotated_v1n = self._rotate_z(point, z_rotation_angle)

            radius = marker_size
            v1p_positions.append(
                Vector3(rotated_v1p[0] + position[0], rotated_v1p[1] + position[1], z + position[2]))
            v1p_radii.append(radius * self._radius_multiplier)
            v1n_positions.append(
                Vector3(rotated_v1n[0] + position[0], rotated_v1n[1] + position[1], z + position[2]))
            v1n_radii.append(radius * self._radius_multiplier)
        
            ax.plot(vx1p, vy1p, marker='o', color='red', markersize=radius)
            ax.plot(vx1n, vy1n, marker='o', color='blue', markersize=radius)
            ax.plot([vx1p, vx1n], [vy1p, vy1n], color='grey', linestyle='--', linewidth=0.5)
            
        status = self._bio_explorer.add_spheres('Proton 1 P', v1p_positions, v1p_radii, Vector3(1, 0, 0))
        status = self._bio_explorer.add_spheres('Proton 1 N', v1n_positions, v1n_radii, Vector3(0, 0, 1))        

    def _load_proton_2(self, ax, sub_particles_to_ignore, marker_size=1.0, collision_maker_size=2.0, j_cut=1.0, output_folder=None, position=[0, 0, 0], z_rotation_angle=0.0):
        i2 = getattr(self._T_collisions, 'i2')
        phi2 = getattr(self._T_collisions, 'phi2')

        v2p_positions = list()
        v2p_radii = list()
        v2n_positions = list()
        v2n_radii = list()

        self._T_protons.GetEntry(i2)
        vxpart2 = np.array(getattr(self._T_protons, 'x'))
        vypart2 = np.array(getattr(self._T_protons, 'y'))
        vzpart2 = np.array(getattr(self._T_protons, 'z'))
        drp2 = np.array(getattr(self._T_protons, 'drp'))
        
        for i in range(self._nb_sub_particles):
            if i in sub_particles_to_ignore:
                continue
            vx2 = self._Rfactor * vxpart2[i] * math.cos(phi2) - self._Rfactor * vypart2[i] * math.sin(phi2);
            vy2 = self._Rfactor * vxpart2[i] * math.sin(phi2) + self._Rfactor * vypart2[i] * math.cos(phi2);
        
            vx2p = vx2 * (1 + drp2)
            vx2n = vx2 * (1 - drp2)
            
            vy2p = vy2 * (1 + drp2)
            vy2n = vy2 * (1 - drp2)

            z = vzpart2[i]

            point = np.array([vx2p, vy2p, z])
            rotated_v2p = self._rotate_z(point, z_rotation_angle)

            point = np.array([vx2n, vy2n, z])  # Example point (1, 0, 0)
            rotated_v2n = self._rotate_z(point, z_rotation_angle)

            radius = marker_size
            v2p_positions.append(
                Vector3(rotated_v2p[0] + position[0], rotated_v2p[1] + position[1], z + position[2]))
            v2p_radii.append(radius * self._radius_multiplier)
            v2n_positions.append(
                Vector3(rotated_v2n[0] + position[0], rotated_v2n[1] + position[1], z + position[2]))
            v2n_radii.append(radius * self._radius_multiplier)
            
            ax.plot(vx2p, vy2p, marker='o', color='magenta', markersize=radius)
            ax.plot(vx2n, vy2n, marker='o', color='cyan', markersize=radius)
            ax.plot([vx2p, vx2n], [vy2p, vy2n], color='grey', linestyle='--', linewidth=0.5)
        
        status = self._bio_explorer.add_spheres('Proton 2 P', v2p_positions, v2p_radii, Vector3(0, 1, 1))
        status = self._bio_explorer.add_spheres('Proton 2 N', v2n_positions, v2n_radii, Vector3(1, 0, 1))

    def _load_collisions(self, ax, marker_size=1.0, collision_maker_size=2.0, j_cut=1.0, output_folder=None):
        njetsCMS = getattr(self._T_collisions, 'njetsCMS')
        ptjetsCMS = getattr(self._T_collisions, 'ptjetsCMS')
        jcaseCMS = getattr(self._T_collisions, 'jcaseCMS')
        xcol = getattr(self._T_collisions, 'xcol')
        ycol = getattr(self._T_collisions, 'ycol')
        icol = getattr(self._T_collisions, 'icol')
        jcol = getattr(self._T_collisions, 'icol')

        # Proton 1 information
        i1 = getattr(self._T_collisions, 'i1')
        self._T_protons.GetEntry(i1)
        vxpart1 = np.array(getattr(self._T_protons, 'x'))
        vypart1 = np.array(getattr(self._T_protons, 'y'))
        proton_1_sub_particle_ids = list()

        # Proton 1 information
        i2 = getattr(self._T_collisions, 'i2')
        self._T_protons.GetEntry(i2)
        vxpart2 = np.array(getattr(self._T_protons, 'x'))
        vypart2 = np.array(getattr(self._T_protons, 'y'))
        proton_2_sub_particle_ids = list()

        col_positions = list()
        col_radii = list()

        nj = 0
        for i in range(njetsCMS):
            if ptjetsCMS[i] < j_cut:
                continue

            x = xcol[i]
            y = ycol[i]
            if x == 0.0 and y == 0:
                continue

            nj += 1
            ms = math.log(ptjetsCMS[i])
            marker_color = 'b'
            if jcaseCMS[i] == 0: 
                marker_color = 'g'
            if jcaseCMS[i] == 1:
                marker_color = 'b'
            if jcaseCMS[i] == 2:
                marker_color = 'r'

            radius = collision_maker_size * ms
            ax.plot(
                xcol[i], ycol[i],
                marker='o', color=marker_color,
                markersize=radius)
            
            col_positions.append(Vector3(x, y, 0.0))
            col_radii.append(radius * self._radius_multiplier)

            proton_1_sub_particle_ids.append(icol[i])
            proton_2_sub_particle_ids.append(jcol[i])
            
        status = self._bio_explorer.add_spheres('Collisions', col_positions, col_radii, Vector3(0, 1, 0))
        return proton_1_sub_particle_ids, proton_2_sub_particle_ids
    
    def _load_particles(self, ax, magnetic, timestamp=1.0):
        nchCMS = getattr(self._T_collisions, 'nchCMS')
        pjet = getattr(self._T_collisions, 'pjet')
        ppx = getattr(self._T_collisions, 'ppx')
        ppy = getattr(self._T_collisions, 'ppy')
        ppz = getattr(self._T_collisions, 'ppz')
        ppt = getattr(self._T_collisions, 'ppt')
        xcol = getattr(self._T_collisions, 'xcol')
        ycol = getattr(self._T_collisions, 'ycol')
        ptype = getattr(self._T_collisions, 'ptype')

        origins = dict()
        targets = dict()
        origins_radii = dict()
        targets_radii = dict()
        colors = dict()
        types = set()

        nchj = np.zeros(300)
        jetphi = np.zeros(300)
        jettheta = np.zeros(300)

        # Loop on particles
        for i in range(nchCMS):
            jet = pjet[i]
            px = ppx[i]
            py = ppy[i]
            pz = ppz[i]
            if pz == 0.0:
                pz = ppt[i]
            
            p = math.sqrt(px * px + py * py + pz * pz)
            lwid = math.log(3 + p) * 0.002

            color = Vector3(1, 1, 1)
            type = ptype[i]
            types.add(type)
            if type == 0:
                color = Vector3(0, 1, 0)
            elif type == 1:
                color = Vector3(0, 0, 1)
            elif type == 2:
                color = Vector3(1, 0, 0)
            elif type > 2:
                color = Vector3(1, 1, 0)
            colors[type] = color

            if jet < 0:
                x1 = 1 * px / p
                y1 = 1 * py / p
                z1 = 1 * pz / p  * timestamp
            else:
                x1 = xcol[jet]
                y1 = ycol[jet]
                z1 = 0.0
                
            nchj[jet] += 1
            jetphi[jet] += math.atan2(py, px)
            jettheta[jet] += math.acos(pz / p)

            if type not in origins:
                origins[type] = list()
            origins[type].append(Vector3(x1, y1, z1))
            
            if type not in origins_radii:
                origins_radii[type] = list()
            origins_radii[type].append(lwid)
        
            plen = math.log(p + 3) / 4
            x2 = x1 + px * plen / p
            y2 = y1 + py * plen / p
            z2 = (z1 + pz * plen / p) * timestamp
            if type not in targets:
                targets[type] = list()
            targets[type].append(Vector3(x2, y2, z2))
            
            if type not in targets_radii:
                targets_radii[type] = list()
            targets_radii[type].append(lwid)

        for t in types:
            if magnetic:
                # self._bio_explorer.add_spheres(
                #     name='Jets origins %d' % t,
                #     positions=origins[t], radii=origins_radii[t],
                #     color=colors[t]
                # )
                self._bio_explorer.add_spheres(
                    name='Jets targets %d' % t,
                    positions=targets[t], radii=targets_radii[t],
                    color=colors[t]
                )
            else:
                for k in range(len(origins[t])):
                    ax.plot(
                        [origins[t][k].x, targets[t][k].x],
                        [origins[t][k].y, targets[t][k].y],
                        color=[colors[t].x, colors[t].y, colors[t].z],
                        markersize=origins_radii[t][k])
                self._bio_explorer.add_cones(
                    name='Line %03d' % t,
                    origins=origins[t], origins_radii=origins_radii[t],
                    targets=targets[t], targets_radii=targets_radii[t],
                    color=colors[t])
        return nchj, jetphi, jettheta
    
    def _load_jets(self, ax, magnetic, nchj, jetphi, jettheta, timestamp):
        xcol = getattr(self._T_collisions, 'xcol')
        ycol = getattr(self._T_collisions, 'ycol')
        njetsCMS = getattr(self._T_collisions, 'njetsCMS')
        ptjetsCMS = getattr(self._T_collisions, 'ptjetsCMS')
        jcaseCMS = getattr(self._T_collisions, 'jcaseCMS')
        phijetsCMS = getattr(self._T_collisions, 'phijetsCMS')
        yjetsCMS = getattr(self._T_collisions, 'yjetsCMS')
        
        corrector = 2.0 # ?? What do lengths have to be multiplied by 2?
        origins = dict()
        targets = dict()
        origins_radii = dict()
        targets_radii = dict()
        colors = dict()
        types = set()

        jphi = 0.0
        jtheta = 0.0
        for j in range(njetsCMS):
            if nchj[j] > 0.0:
                jphi = jetphi[j] / nchj[j]
                jtheta = jettheta[j] / nchj[j]

            r = ptjetsCMS[j]
            if r <= 0.0:
                continue
            rjet = math.log(r) / 3.0
            if rjet <= 0.0:
                continue
                
            type = jcaseCMS[j]
            types.add(type)
            if type == 0:
                color = Vector3(0, 1, 0)
            elif type == 1:
                color = Vector3(0, 0, 1)
            elif type == 2:
                color = Vector3(1, 0, 0)
            colors[type] = color

            phi = phijetsCMS[j]
            theta = 2.0 * math.atan(math.exp(-yjetsCMS[j]))

            '''Origins'''
            l = 0.5 * corrector * timestamp
            x1 = xcol[j] + l * rjet * math.sin(jtheta) * math.cos(jphi);
            y1 = ycol[j] + l * rjet * math.sin(jtheta) * math.sin(jphi);
            z1 = l * rjet * math.cos(jtheta);
            
            '''Targets'''
            jphi *= 180.0 / math.pi
            jtheta *= 180.0 / math.pi
            phi *= 180.0 / math.pi
            theta *= 180.0 / math.pi
            
            cone_vector = np.array([0.0, 0.0, -1.0]) # Invert Z for BioExplorer
            v = self._rotate_vector(
                cone_vector,
                math.radians(-jtheta),
                0.0,
                math.radians(jphi - 90.0),
            )
            radius = 0.5 * rjet * corrector * timestamp
            v[0] *= radius
            v[1] *= radius
            v[2] *= radius

            if type not in origins:
                origins[type] = list()
            origins[type].append(Vector3(x1, y1, z1))
                
            if type not in origins_radii:
                origins_radii[type] = list()
            origins_radii[type].append(0.1 * rjet)
            
            if type not in targets:
                targets[type] = list()
            if type==2: # Why?!?!?
                targets[type].append(Vector3())
            else:
                targets[type].append(Vector3(x1 + v[0], y1 + v[1], z1 + v[2]))
            
            if type not in targets_radii:
                targets_radii[type] = list()
            targets_radii[type].append(0.0)

        for t in types:
            if magnetic:
                # self._bio_explorer.add_spheres(
                #     name='Jets origins %d' % t,
                #     positions=origins[t], radii=origins_radii[t],
                #     color=colors[t]
                # )
                self._bio_explorer.add_spheres(
                    name='Jets targets %d' % t,
                    positions=targets[t], radii=targets_radii[t],
                    color=colors[t]
                )
            else:
                self._bio_explorer.add_cones(
                    name='Jets %d' % t,
                    origins=origins[t], origins_radii=origins_radii[t],
                    targets=targets[t], targets_radii=targets_radii[t],
                    color=colors[t], opacity=0.3)

    def plot_event(self, event_id, colormap=None, magnetic=False, timestamp=0.0, voxel_size=1.0, value_range=[0.0, 0.02], export_filename=None, marker_size=1.0, show_grid=False):
        self._T_collisions.GetEntry(event_id)

        status = self._bio_explorer.reset_scene()
        fig, ax = plt.subplots(figsize=(6,6))

        x2offset = getattr(self._T_collisions, 'x2offset')
        y2offset = getattr(self._T_collisions, 'y2offset')
        
        rotation_speed = timestamp * 0.06 # 6% of light speed

        proton_1_sub_particle_ids = list()
        proton_2_sub_particle_ids = list()
        if not magnetic:
            if show_grid:
                self._bio_explorer.add_grid(min_value=-5.0, max_value=5.0, interval=0.5, radius=0.001, show_planes=True, opacity=0.2)
            
            if timestamp>=0:
                proton_1_sub_particle_ids, proton_2_sub_particle_ids = self._load_collisions(ax)

            if False: # Show protons : TODO Add SDF pancake!!
                corrector = 1.5 # Why??
                self._bio_explorer.add_spheres(name='Proton 1', positions=[Vector3(0, 0, -timestamp)], radii=[corrector * 0.876 * 1.3], color=Vector3(0.5, 0.5, 0.0), opacity=0.25)
                self._bio_explorer.add_spheres(name='Proton 2', positions=[Vector3(x2offset, y2offset, timestamp)], radii=[corrector * 0.876 * 1.3], color=Vector3(0.0, 0.0, 0.5), opacity=0.25)

        self._load_proton_1(
            ax, proton_1_sub_particle_ids, marker_size=marker_size, position=[0,0,-timestamp],
            z_rotation_angle=timestamp * rotation_speed)
        self._load_proton_2(
            ax, proton_2_sub_particle_ids, marker_size=marker_size, position=[0,0, timestamp],
            z_rotation_angle=-timestamp * rotation_speed)

        if timestamp>=0:
            nchj, jetphi, jettheta = self._load_particles(ax, magnetic, timestamp)
            self._load_jets(ax, magnetic, nchj, jetphi, jettheta, timestamp)

        if magnetic:
            '''Scene bounds'''
            self._bio_explorer.add_spheres(
                name='Bounds',
                positions=[
                    Vector3(-4.493167877197266, -3.151878595352173, -3.7470436096191406),
                    Vector3(4.370408058166504, 3.2409679889678955, 4.623395919799805),
                ],
                radii=[0.0001, 0.0001])

            status = self._bio_explorer.build_fields(voxel_size=voxel_size)
            model_id = self._bio_explorer.get_model_ids()['ids'][-1:][0]
            tf = TransferFunction(
                bioexplorer=self._bio_explorer, model_id=model_id,
                filename=colormap,
                value_range=value_range,
            )
            model_ids = self._bio_explorer.get_model_ids()['ids'][:-1]
            for model_id in model_ids:
                self._core.update_model(id=model_id, visible=False)
            
        if export_filename:
            plt.savefig(export_filename)
        plt.show()

    def set_field_parameters(self, cutoff_distance=1e6, sampling_rate=1.0, gradient_shading=False, gradient_offset=0.01, epsilon=0.25, accumulation_steps=0, use_octree=True):
        self._core.set_field_parameters(
            cutoff=cutoff_distance, sampling_rate=sampling_rate,
            gradient_shading=gradient_shading, gradient_offset=gradient_offset,
            epsilon=epsilon, accumulation_steps=accumulation_steps, use_octree=use_octree
        )
        self._core.set_renderer()        