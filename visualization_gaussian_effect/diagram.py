import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def generate_numbers(p,n):
    k = 30
    x = np.random.uniform(0,1,(n,k))
    x = (x>=p)*1
    out = np.zeros(n)
    for i in range(n):
        out[i] = int(''.join(map(lambda b: str(int(b)), x[i])), 2)/(2**k)
    return out



def f(sources, k=2.5):
    x = np.linspace(-1,1,num=1000)
    y = np.linspace(-1,1,num=1000)
    X, Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    for x_point, y_point in sources:
        X -= x_point
        Y -= y_point
        Z += k*np.e**(-k*k*(X*X + Y*Y))

        X += x_point
        Y += y_point

    return X,Y,Z


def f_sample_values(sources,samplig_points,k=2.5):
    x, y = zip(*samplig_points)
    X,Y = np.array(x), np.array(y)
    Z = np.zeros_like(X)
   
    for x_point, y_point in sources:
        X -= x_point
        Y -= y_point
        Z += k*np.e**(-k*k*(X*X + Y*Y))

        X += x_point
        Y += y_point

    return Z

def gradient_f(sources,network_samples,k=2.5):
    x = np.linspace(-1,1,num=50)
    y = np.linspace(-1,1,num=50)
    X, Y = np.meshgrid(x,y)

    U = np.zeros_like(X)
    V = np.zeros_like(X)

    ps = f_sample_values(network_samples,sources,k=k)
    print(ps)
    i = 0

    for x_point, y_point in sources:
        X -= x_point
        Y -= y_point
        U += -k*k*np.e**(-k*k*(X*X + Y*Y))*2*X/ps[i]
        V += -k*k*np.e**(-k*k*(X*X + Y*Y))*2*Y/ps[i]

        X += x_point
        Y += y_point

        i+=1

    return X,Y,U,V




def gradient_f_sampled(sources,network_samples,k=2.5, sum_to_see=None):

    if sum_to_see is None:
        sum_to_see = sources

    X, Y = zip(*network_samples)
    X, Y = np.array(X), np.array(Y)
    U = np.zeros_like(X)
    V = np.zeros_like(X)

    ps = f_sample_values(network_samples,sources,k=k)
    i = 0
    for x_point, y_point in sum_to_see:
        X -= x_point
        Y -= y_point
        U += -k*k*np.e**(-k*k*(X*X + Y*Y))*2*X/ps[i]
        V += -k*k*np.e**(-k*k*(X*X + Y*Y))*2*Y/ps[i]

        X += x_point
        Y += y_point

        i+=1

    return U,V



network_points = [(.25,0), (-0.5,0.5), (-0.5,-0.5) ]
network_points = [(.25,0), (-0.5,0.6), (-0.5,-0.5) ]

fig, ax = plt.subplots(1,2,sharey=True, sharex=True,figsize=(12,12), subplot_kw=dict(box_aspect=1))
s = 300

k=5.5

X, Y, Z = f(network_points,k=k)
# im = ax.imshow(Z,aspect="auto", alpha=0.8,  vmin=0,vmax=3,cmap="Blues", interpolation="bicubic", extent=[-1,1,-1,1])
ax[0].contourf(X, Y, Z, alpha=0.4) 


ax[0].scatter(*zip(*network_points),color="#6699ff", edgecolors='#0000ff' , s=s)


# dataset_points = [-0.2, -0.40],[.1,-0.25]
dataset_points = [(-0.2,.1) ,(-0.40,-.25)]

ax[0].scatter(*zip(*dataset_points), color="#ff0000", edgecolors='#990000', s=s)

# plt.text(0,-0.5, 'thumbnails sample', fontsize = 11)
# plt.text(.25,0.25, 'network sample', fontsize = 11)
arrowprops = dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.4",
                            color='k')
ax[0].annotate("network sample",fontsize=20, xy=(0.35,0), xytext=(.15,0.35), 
    arrowprops=arrowprops)
                            
ax[0].annotate("thumbnails sample",fontsize=20, xy=(-0.4,-0.30),xytext=(0,-0.75), xycoords='data',arrowprops=arrowprops)

ax[0].grid(False)
ax[0].axis('off')




X, Y, Z = f(dataset_points, k=k)
# ax[1].contourf(X, Y, Z, alpha=0.4, cmap="jet") 

X, Y, U, V = gradient_f(dataset_points, network_points, k=k)
# ax[1].quiver(X, Y, U/2, V/2, alpha = 0.4)

ax[1].scatter(*zip(*dataset_points), color="#ff0000", edgecolors='#990000', s=s)
ax[1].scatter(*zip(*network_points),color="#6699ff", edgecolors='#0000ff' , s=s)
arrow_scale = 0.12


# for dataset_point in dataset_points:

#     arrows = gradient_f_sampled(dataset_points,network_points,sum_to_see=[dataset_point],k=k)
#     arrows = arrows[0]*arrow_scale,  arrows[1]*arrow_scale
#     for i, point in enumerate(network_points):
#         ax[1].annotate("", xytext=point, xy=(point[0]+arrows[0][i], point[1]+arrows[1][i]),arrowprops=dict(arrowstyle="->", color="red"))
#     # plt.savefig("gaussian_effect.svg",  bbox_inches='tight')

arrows = gradient_f_sampled(dataset_points,network_points,k=k)
arrows = arrows[0]*arrow_scale,  arrows[1]*arrow_scale

for i, point in enumerate(network_points):
    ax[1].annotate("", xytext=point, xy=(point[0]+arrows[0][i], point[1]+arrows[1][i]),arrowprops=dict(arrowstyle="->", color="k"))

 
ax[1].grid(True)

# plt.grid(False)
plt.tight_layout()
plt.axis('off')

plt.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0.062,hspace=0)
# plt.savefig("gaussian_effect2.svg",  bbox_inches='tight')

plt.show()

