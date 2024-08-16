from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('pgf')
import pandas as pd
import tikzplotlib
import numpy as np
from sklweka.dataset import load_arff, to_nominal_labels
from scipy.optimize import fsolve
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import mahalanobis
import settings
import os
from  matplotlib.colors import LinearSegmentedColormap


plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble' : "\n".join([r'\usepackage{amsmath}',r'\usepackage{amssymb}']),
    'pgf.preamble' : "\n".join([r'\usepackage{amsmath}',r'\usepackage{amssymb}',r"\usepackage{bm}"]),
    'pgf.rcfonts': False,
    'pgf.texsystem': "pdflatex",
})


def prepare_example(input_file_path, output_directory):
    os.makedirs(output_directory,exist_ok=True)
    X, y, meta = load_arff(input_file_path, class_index="last")
    y = y.astype(np.int8)

    x_min, x_max = np.min(X[:,0]), np.max(X[:,0])
    y_min, y_max = np.min(X[:,1]), np.max(X[:,1])

    y_mult = 0.5
    class_1 = y == 0
    class_2 = y == 1
    x = np.linspace(-0.1,1.1,1000)

    out_pdf_file_path = os.path.join(output_directory, "example1.pdf")
    single_pdf_path = os.path.join(output_directory, "example1_s")
    os.makedirs(single_pdf_path, exist_ok=True)

    my_cmap =LinearSegmentedColormap.from_list('rg',["r", "g"], N=256) 
    
    with PdfPages(out_pdf_file_path) as pdf:

        # Decision boundary

        c_2_1 = np.asanyarray([0.1,0.6])
        c_2_2 = np.asanyarray([0.65,0.5])

        c_1_1 = np.asanyarray([0.4,0.4])
        c_1_2 = np.asanyarray([0.9,0.5])

        clusters= [(c_1_1,0), (c_1_2,0), (c_2_1, 1), (c_2_2, 1)]
        plt.figure(figsize=(8, 6))

        plt.scatter(X[class_1,0], X[class_1,1],color="red", marker='+', label="class 1")
        plt.scatter(X[class_2,0], X[class_2,1],color="green", marker='x', label="class 2")
        #Starting point on the line y = 0.5x
        start_point = np.array([0.3, 0.15])

        # Perpendicular vector direction
        direction = np.array([1, -2])

        # Normalize the direction vector to length 1
        unit_vector = direction / np.linalg.norm(direction)

        # Scale to length of 1
        vector_length = 0.25
        scaled_vector = unit_vector * vector_length

        # Add the vector to the plot
        plt.arrow(start_point[0], start_point[1],  -scaled_vector[0],  -scaled_vector[1], 
                head_width=0.01, head_length=0.02, fc='black', ec='black')
        plt.text(start_point[0]  -scaled_vector[0] / 2, start_point[1] - scaled_vector[1] / 2, '$\\boldsymbol{n}$', fontsize=12, color='black')

        for cluster_centroid, sign in clusters:

            marker = "v" if sign == 1 else "s"
            plt.scatter(cluster_centroid[0], cluster_centroid[1], color="black", marker=marker, s=60)

        plt.plot(x,y_mult*x, color="black")
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.title("Decision Boundary")
        plt.xlabel('$x_1$')
        plt.ylabel("$x_2$")
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        data_pdf_path = os.path.join(single_pdf_path, "example_data.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()

        # Decision boundary potential

        dec_vec = np.asanyarray([-1*y_mult,1])

        c1_dists = np.dot(X, dec_vec)
        c1_dists = c1_dists[c1_dists<=0]

        c2_dists = np.dot(X, dec_vec)
        c2_dists = c2_dists[c2_dists>=0]

        ## distances histogram
        plt.hist(c1_dists,color='red',alpha=0.5, density=True, label="class 1")
        plt.hist(c2_dists, color='green', alpha=0.5, density=True, label="class 2")
        # plt.title("Decision function values histogram")
        plt.xlabel("Values of the decision function")
        plt.ylabel("Probability density")
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        data_pdf_path = os.path.join(single_pdf_path, "example_hist.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()

        c1_quantile = np.quantile(c1_dists, [0.1])
        c2_quantile = np.quantile(c2_dists, [0.9])

        def f1(a):
            return np.tanh(a* c1_quantile) - 0.9
        
        def f2(a):
            return np.tanh(a* c2_quantile) - 0.9
        
        a1 = fsolve(f1, 1.0)
        a2 = fsolve(f2, 1.0)

        print("A1:", a1, "A2:", a2)

        def t1(x):
        
            ret = -np.tanh(x * a1)
            ret[ret>=0] = 0
            return ret
        
        def t2(x):
            ret = np.tanh(x * a2)
            ret[ret<0] = 0
            return ret
        
        def bnd_potential(grid):
            di = np.tensordot(grid, dec_vec,axes=1)
            ret = t1(di) + t2(di)
            return ret


        x = np.linspace(np.min(c1_dists), np.max(c2_dists),1000)
        x1 = np.linspace(np.min(c1_dists), 0,1000)
        x2 = np.linspace(0, np.max(c2_dists),1000)
        # plt.plot(x,t1(x)+t2(x), color='black')
        plt.plot(x1, t1(x1), color='red', label="class 1", linestyle='solid')
        plt.plot(x2, t2(x2), color='green', label="class 2", linestyle='dashed')
        # plt.title("Assymetric Potential")
        plt.xlabel("Values of the decision function")
        plt.ylabel("Scoring value")
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_lin_2d.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()

        ## plot 3d mesh

        x_ax = np.linspace(x_min, x_max, 100)
        y_ax = np.linspace(y_min, y_max, 100)
        xg, yg = np.meshgrid(x_ax, y_ax)
        grid =np.dstack((xg,yg))
        bnd_p = bnd_potential(grid)

        # my_cmap = plt.get_cmap('hot')
        fig, ax = plt.subplots(subplot_kw={"projection": "3d", "elev":30, "azim":210,})
        
        surf = ax.plot_surface(xg,yg,bnd_p, edgecolors='k',lw=0.6, shade=True, alpha=0.3,
                                cmap=my_cmap, cstride=3, rstride=3)
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(r'$p^{(i)}(\boldsymbol{x})$')
        # ax.set_title("Boundary potential function")
        cbaxes = fig.add_axes([0.81, 0.2, 0.01, 0.3])
        fig.colorbar(surf, cax = cbaxes,
             shrink = 0.2, aspect = 10, use_gridspec=True)
        pdf.savefig(bbox_inches='tight')
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_lin_3d.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()

        cp = plt.contourf(xg, yg, bnd_p, cmap=my_cmap, alpha=0.95, levels=20)
        plt.colorbar(cp, label=r'$p^{(i)}(\boldsymbol{x})$')
        x_line = np.linspace(0, 1, 100)
        y_line = 0.5 * x_line
        plt.plot(x_line, y_line, color='black', linestyle='solid', linewidth=2,)
        # plt.title('Contour plot of Z values')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        pdf.savefig(bbox_inches='tight')
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_lin_2d_h.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()

        # Cluster potentials
        po = PotentialF()
        po.fit(X,y,clusters)

        ## plot 3d mesh

        x_ax = np.linspace(x_min, x_max, 100)
        y_ax = np.linspace(y_min, y_max, 100)
        xg, yg = np.meshgrid(x_ax, y_ax)
        grid =np.dstack((xg,yg))

        g= grid.reshape( (100*100,2) )

        z= po.potential(g)
        zp = po.potential2(g)
        z = z.reshape((100,100))
        zp = zp.reshape((100,100))
        # my_cmap = plt.get_cmap('hot')
        fig, ax = plt.subplots(subplot_kw={"projection": "3d", "elev":30, "azim":260,})
        
        surf = ax.plot_surface(xg,yg,z, edgecolors='k',lw=0.6, shade=True, alpha=0.3,
                                cmap=my_cmap, cstride=3, rstride=3)
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(r'$\mu^{(i)}_{m}(\boldsymbol{x})$')
        # ax.set_title("Cluster potential function")
        cbaxes = fig.add_axes([0.78, 0.2, 0.01, 0.3])
        fig.colorbar(surf, cax = cbaxes,
             shrink = 0.2, aspect = 10, use_gridspec=True)
        pdf.savefig(bbox_inches='tight')
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_cluster_3d.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()
        
        cp = plt.contourf(xg, yg, z, cmap=my_cmap, alpha=0.95, levels=15)
        zero_contour = plt.contour(xg, yg, z, levels=[0], colors='black', linewidths=2)
        plt.colorbar(cp, label=r'$\mu^{(i)}_{m}(\boldsymbol{x})$')
        x_line = np.linspace(0, 1, 100)
        y_line = 0.5 * x_line
        plt.plot(x_line, y_line, color='darkgray', linestyle='dashed', linewidth=2,)
        # plt.title('Contour plot of Z values')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        pdf.savefig(bbox_inches='tight')
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_cluster_2d.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()

        cp = plt.contourf(xg, yg, zp, cmap=my_cmap, alpha=0.95, levels=15)
        zero_contour = plt.contour(xg, yg, zp, levels=[0], colors='black', linewidths=2)
        plt.colorbar(cp, label=r'$\mu^{(i)}_{m}(\boldsymbol{x})$')
        x_line = np.linspace(0, 1, 100)
        y_line = 0.5 * x_line
        plt.plot(x_line, y_line, color='darkgray', linestyle='dashed', linewidth=2,)
        # plt.title('Contour plot of Z values')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        pdf.savefig(bbox_inches='tight')
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_cluster_2d_a.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()
        
        # overall potential
        
        # my_cmap = plt.get_cmap('hot')
        fig, ax = plt.subplots(subplot_kw={"projection": "3d", "elev":30, "azim":240,})
        
        surf = ax.plot_surface(xg,yg,0.5*(z+bnd_p), edgecolors='k',lw=0.6, shade=True, alpha=0.3,
                                cmap=my_cmap, cstride=3, rstride=3)
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(r'$\nu^{(i)}_{m}(\boldsymbol{x})$')
        # ax.set_title("Cumulated potential function")
        cbaxes = fig.add_axes([0.79, 0.2, 0.01, 0.3])
        fig.colorbar(surf, cax = cbaxes,
             shrink = 0.2, aspect = 10, use_gridspec=True)
        pdf.savefig(bbox_inches='tight')
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_overall_3d.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()

        
        cp = plt.contourf(xg, yg, 0.5*(z+bnd_p), cmap=my_cmap, alpha=0.85, levels=20)
        zero_contour = plt.contour(xg, yg, 0.5*(z+bnd_p), levels=[0], colors='black', linewidths=2)
        x_line = np.linspace(0, 1, 100)
        y_line = 0.5 * x_line
        plt.plot(x_line, y_line, color='darkgray', linestyle='dashed', linewidth=2,)
        plt.colorbar(cp, label=r'$\nu^{(i)}_{m}(\boldsymbol{x})$')
        # plt.title('Contour plot of Z values')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        pdf.savefig()
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_2d.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.scatter(X[class_1,0], X[class_1,1],color="darkred", marker='+', label="class 1")
        plt.scatter(X[class_2,0], X[class_2,1],color="darkgreen", marker='x', label="class 2")
        pdf.savefig()
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_2d_d.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()


        funnn = 0.7*z+ 0.3*bnd_p
        cp = plt.contourf(xg, yg, funnn , cmap=my_cmap, alpha=0.85, levels=20)
        zero_contour = plt.contour(xg, yg, funnn, levels=[0], colors='black', linewidths=2)
        x_line = np.linspace(0, 1, 100)
        y_line = 0.5 * x_line
        plt.plot(x_line, y_line, color='darkgray', linestyle='dashed', linewidth=2,)
        plt.colorbar(cp, label=r'$\nu^{(i)}_{m}(\boldsymbol{x})$')
        # plt.title('Contour plot of Z values')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        pdf.savefig()
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_2d_2.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.scatter(X[class_1,0], X[class_1,1],color="darkred", marker='+', label="class 1")
        plt.scatter(X[class_2,0], X[class_2,1],color="darkgreen", marker='x', label="class 2")
        pdf.savefig()
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_2d_2_d.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()

        funnn = 0.9*z+ 0.1*bnd_p
        cp = plt.contourf(xg, yg, funnn , cmap=my_cmap, alpha=0.85, levels=20)
        zero_contour = plt.contour(xg, yg, funnn, levels=[0], colors='black', linewidths=2)
        x_line = np.linspace(0, 1, 100)
        y_line = 0.5 * x_line
        plt.plot(x_line, y_line, color='darkgray', linestyle='dashed', linewidth=2,)
        plt.colorbar(cp, label=r'$\nu^{(i)}_{m}(\boldsymbol{x})$')
        # plt.title('Contour plot of Z values')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        pdf.savefig()
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_2d_3.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.scatter(X[class_1,0], X[class_1,1],color="darkred", marker='+', label="class 1")
        plt.scatter(X[class_2,0], X[class_2,1],color="darkgreen", marker='x', label="class 2")
        pdf.savefig()
        data_pdf_path = os.path.join(single_pdf_path, "example_potential_2d_3_d.pdf")
        plt.savefig(data_pdf_path, format="pdf",bbox_inches='tight' )
        plt.close()

        

class PotentialF:


    def potential(self, X):

        n_clusters = len(self._cluster_np)
        cl_dists_potentials = np.zeros((n_clusters, len(X)))
        for cl_idx in range(n_clusters):
            mah_dist =  np.diag( np.sqrt(np.dot(np.dot((self._cluster_np[cl_idx] - X), self._cluster_i_covs[cl_idx]), (self._cluster_np[cl_idx] - X).T)))
            cl_dists_potentials[cl_idx] = (1 - np.tanh(mah_dist * self._alphas[cl_idx]))
            if self._cluster_class[cl_idx]== 0:
                cl_dists_potentials[cl_idx]*=-1

        avg_pot = np.mean(cl_dists_potentials,axis=0)
        return avg_pot
    
    def potential2(self, X):

        n_clusters = len(self._cluster_np)
        cl_dists_potentials = np.zeros((n_clusters, len(X)))
        for cl_idx in range(n_clusters):
            mah_dist =  np.diag( np.sqrt(np.dot(np.dot((self._cluster_np[cl_idx] - X), self._cluster_i_covs[cl_idx]), (self._cluster_np[cl_idx] - X).T)))
            cl_dists_potentials[cl_idx] = (1 - np.tanh(mah_dist * self._alphas[cl_idx]))
            if self._cluster_class[cl_idx]== 0:
                cl_dists_potentials[cl_idx]*=-1

        abs_pot = np.abs(cl_dists_potentials)
        max_indices = np.argmax(abs_pot,axis=0)

        result = cl_dists_potentials[max_indices, np.arange(cl_dists_potentials.shape[1])]
        return result
    
     

    def fit(self, X, y, c):
        self._cluster_np = np.asanyarray( [ cl for cl, si in c])
        self._cluster_class = np.asanyarray( [ si for cl, si in c])
        n_clusters = len(c)

        train_class_idx = np.asanyarray([y==0, y==1])
        cluster_class_idx = np.asanyarray( [self._cluster_class ==0, self._cluster_class == 1 ])

        self._cluster_i_covs = np.zeros((n_clusters,2,2))
        self._alphas = np.ones( (n_clusters,) )
        np_all_dists = euclidean_distances(self._cluster_np, X) # n_clusters x n_points

        for cl_idx, (cl, si) in enumerate(c):
            
            dis_mod = np_all_dists.copy()
            dis_mod[np.logical_not( cluster_class_idx[si] ),:] = np.inf
            cluster_selected = np.logical_and(np.argmin(dis_mod,axis=0) == cl_idx, train_class_idx[si])
            cov_m = np.cov(X[cluster_selected],rowvar=False)
            self._cluster_i_covs[cl_idx] = np.linalg.inv(cov_m)

            mah_dist =  np.diag( np.sqrt(np.dot(np.dot((cl - X[cluster_selected]), self._cluster_i_covs[cl_idx]), (cl - X[cluster_selected]).T)))
            q = np.quantile(mah_dist, 0.9)
            
            def f1(a):
                return np.tanh(a* q) - 0.9
        
            a1 = fsolve(f1, 1.0)
            self._alphas[cl_idx] = a1
        print("Alphas:", self._alphas)
        

if __name__ == '__main__':

    data_dir = settings.DATAPATH
    input_file_path = os.path.join(data_dir, "Banana_2d_STD.arff")
    outputdir = os.path.join(settings.RESULSTPATH, "Example1")

    prepare_example(input_file_path, outputdir)
