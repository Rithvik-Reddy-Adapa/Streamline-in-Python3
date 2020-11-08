'''
    Streamline in Python3
    Copyright (C) Year: 2020,  Author: Rithvik Reddy Adapa

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

def streamline(*,x,y,z=None,u,v,w=None,xi,yi,zi=None,n,length,r_mag=False,r_u=False,r_v=False,r_w=False):
    '''
    #   Made on 23 / 07 / 2020 (dd / mm / yyyy)
    #   modules to install "numpy, scipy"
    #    x,y,z,u,v,w = given vector data (u,v,w) for corresponding (x,y,z)
    #       x,y,z,u,v,w should be 3d numpy array ( for 3d streamline )
    #       x,y,z should be grided as "[y,x,z]=numpy.meshgrid(y,x,z)"
    #                                      OR
    #       x,y,u,v should be 2d numpy array ( for 2d streamline )
    #       x,y should be grided as "[y,x]=numpy.meshgrid(y,x)"
    #   xi = initial x values as a row i.e. 1d row numpy array
    #   yi = initial y values as a row i.e. 1d row numpy array
    #   zi = initial z values as a row i.e. 1d row numpy array
    #   n = total number of lines = total number of points (excluding initial point)
    #   length = length of each line = distance between consecutive points
    #   r_mag(return magnitude) = whether to return magnitude of vectors or not       |   all these
    #   r_u(return u) = whether to return i_cap(x component) of vectors or not          | are useful
    #   r_v(return v) = whether to return j_cap(y component) of vectors or not          |  for
    #   r_w(return w) = whether to return k_cap(z component) of vectors or not        |   gradient plotting (i.e. useful to color the lines based on these values)
    #   returns in the following format:
    #       [xo,yo,zo,(mag),(i_cap),(j_cap),(k_cap)]
    #           mag is returned if r_mag==True
    #           i_cap is returned if r_u==True
    #           j_cap is returned if r_v==True
    #           k_cap is returned if r_w==True
    #       xo, yo,zo,mag,i_cap,j_cap,k_cap have same dimensions i.e. array shape = (xi.shape,n+1)
    '''
    
    from scipy.interpolate import interpn
    import numpy as np

    if np.all(z!=None) and z[0,0,:].shape[0]>1: # 3d streamline
        
        xo=np.zeros([xi.shape[-1],n+1])
        yo=np.zeros([yi.shape[-1],n+1])
        zo=np.zeros([zi.shape[-1],n+1])
        if r_mag==True: mag=np.zeros([xi.shape[-1],n+1])
        if r_u==True: i_cap=np.zeros([xi.shape[-1],n+1])
        if r_v==True: j_cap=np.zeros([xi.shape[-1],n+1])
        if r_w==True: k_cap=np.zeros([xi.shape[-1],n+1])

        xo[:,0]=xi
        yo[:,0]=yi
        zo[:,0]=zi

        for j in np.arange(xi.shape[-1]):
            for i in np.arange(n):
                
                a=interpn((x[:,0,0],y[0,:,0],z[0,0,:]),u,[xo[j,i],yo[j,i],zo[j,i]],bounds_error=False)
                b=interpn((x[:,0,0],y[0,:,0],z[0,0,:]),v,[xo[j,i],yo[j,i],zo[j,i]],bounds_error=False)
                c=interpn((x[:,0,0],y[0,:,0],z[0,0,:]),w,[xo[j,i],yo[j,i],zo[j,i]],bounds_error=False)

                if r_mag==True: mag[j,i]=(a**2+b**2+c**2)**(1/2)
                if r_u==True: i_cap[j,i]=a
                if r_v==True: j_cap[j,i]=b
                if r_w==True: k_cap[j,i]=c

                xo[j,i+1]=xo[j,i]+(a*length)/((a**2+b**2+c**2)**(1/2))
                yo[j,i+1]=yo[j,i]+(b*length)/((a**2+b**2+c**2)**(1/2))
                zo[j,i+1]=zo[j,i]+(c*length)/((a**2+b**2+c**2)**(1/2))
            
            a=interpn((x[:,0,0],y[0,:,0],z[0,0,:]),u,[xo[j,n],yo[j,n],zo[j,n]],bounds_error=False)
            b=interpn((x[:,0,0],y[0,:,0],z[0,0,:]),v,[xo[j,n],yo[j,n],zo[j,n]],bounds_error=False)
            c=interpn((x[:,0,0],y[0,:,0],z[0,0,:]),w,[xo[j,n],yo[j,n],zo[j,n]],bounds_error=False)

            if r_mag==True: mag[j,n]=(a**2+b**2+c**2)**(1/2)
            if r_u==True: i_cap[j,n]=a
            if r_v==True: j_cap[j,n]=b
            if r_w==True: k_cap[j,n]=c

        ret=[xo,yo,zo]
        if r_mag==True: ret.append(mag)
        if r_u==True: ret.append(i_cap)
        if r_v==True: ret.append(j_cap)
        if r_w==True: ret.append(k_cap)
        
        return ret
    
    else: # 2d streamline
        
        del zi # removing unnecessary variable
        if np.all(z!=None) and z[0,0,:].shape[0]==1: x=x[:,:,0]; y=y[:,:,0]; u=u[:,:,0]; v=v[:,:,0];
        
        xo=np.zeros([xi.shape[-1],n+1])
        yo=np.zeros([yi.shape[-1],n+1])
        if r_mag==True: mag=np.zeros([xi.shape[-1],n+1])
        if r_u==True: i_cap=np.zeros([xi.shape[-1],n+1])
        if r_v==True: j_cap=np.zeros([xi.shape[-1],n+1])

        xo[:,0]=xi
        yo[:,0]=yi

        for j in np.arange(xi.shape[-1]):
            for i in np.arange(n):
                
                a=interpn((x[:,0],y[0,:]),u,[xo[j,i],yo[j,i]],bounds_error=False)
                b=interpn((x[:,0],y[0,:]),v,[xo[j,i],yo[j,i]],bounds_error=False)

                if r_mag==True: mag[j,i]=(a**2+b**2+c**2)**(1/2)
                if r_u==True: i_cap[j,i]=a
                if r_v==True: j_cap[j,i]=b

                xo[j,i+1]=xo[j,i]+(a*length)/((a**2+b**2)**(1/2))
                yo[j,i+1]=yo[j,i]+(b*length)/((a**2+b**2)**(1/2))
            
            a=interpn((x[:,0],y[0,:]),u,[xo[j,n],yo[j,n]],bounds_error=False)
            b=interpn((x[:,0],y[0,:]),v,[xo[j,n],yo[j,n]],bounds_error=False)

            if r_mag==True: mag[j,n]=(a**2+b**2)**(1/2)
            if r_u==True: i_cap[j,n]=a
            if r_v==True: j_cap[j,n]=b

        ret=[xo,yo]
        if r_mag==True: ret.append(mag)
        if r_u==True: ret.append(i_cap)
        if r_v==True: ret.append(j_cap)
        
        return ret
