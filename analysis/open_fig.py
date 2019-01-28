from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def plotFig(filename,fignr=1):
   d = loadmat(filename,squeeze_me=True, struct_as_record=False)
   matfig = d['hgS_070000']
   childs = matfig.children
   ax1 = [c for c in childs if c.type == 'axes']
   if(len(ax1) > 0):
       ax1 = ax1[0]
   legs = [c for c in childs if c.type == 'scribe.legend']
   if(len(legs) > 0):
       legs = legs[0]
   else:
       legs=0
   pos = matfig.properties.Position
   size = np.array([pos[2]-pos[0],pos[3]-pos[1]])/96
   plt.figure(fignr,figsize=size)
   plt.clf()
   # plt.hold(True)
   counter = 0
   for line in ax1.children:
       if line.type == 'graph2d.lineseries':
           if hasattr(line.properties,'Marker'):
               mark = "%s" % line.properties.Marker
               if(mark != "none"):
                   mark = mark[0]
           else:
               mark = '.'
           if hasattr(line.properties,'LineStyle'):
               linestyle = "%s" % line.properties.LineStyle
           else:
               linestyle = '-'
           if hasattr(line.properties,'Color'):
               r,g,b =  line.properties.Color
           else:
               r = 0
               g = 0
               b = 1
           if hasattr(line.properties,'MarkerSize'):
               marker_size = line.properties.MarkerSize
           else:
               marker_size = -1
           x = line.properties.XData
           y = line.properties.YData
           if(mark == "none"):
               plt.plot(x,y,linestyle=linestyle,color=[r,g,b])
           elif(marker_size==-1):
               plt.plot(x,y,marker=mark,linestyle=linestyle,color=[r,g,b])
           else:
               plt.plot(x,y,marker=mark,linestyle=linestyle,color=[r,g,b],ms=marker_size)
       elif line.type == 'text':
           # if counter == 0:
               # plt.xlabel("$%s$" % line.properties.String,fontsize =16)
           # if counter == 1:
               # plt.ylabel("$%s$" % line.properties.String,fontsize = 16)
           if counter == 3:
               plt.title("$%s$" % line.properties.String,fontsize = 16)
           counter += 1
   # plt.grid(ax1.properties.XGrid)

   if(hasattr(ax1.properties,'XTick')):
       if(hasattr(ax1.properties,'XTickLabelRotation')):
           plt.xticks(ax1.properties.XTick,ax1.properties.XTickLabel,rotation=ax1.properties.XTickLabelRotation)
       else:
           plt.xticks(ax1.properties.XTick,ax1.properties.XTickLabel)
   if(hasattr(ax1.properties,'YTick')):
       if(hasattr(ax1.properties,'YTickLabelRotation')):
           plt.yticks(ax1.properties.YTick,ax1.properties.YTickLabel,rotation=ax1.properties.YTickLabelRotation)
       else:
           plt.yticks(ax1.properties.YTick,ax1.properties.YTickLabel)
   plt.xlim(ax1.properties.XLim)
   plt.ylim(ax1.properties.YLim)
   if legs:
       leg_entries = tuple(['$' + l + '$' for l in legs.properties.String])
       py_locs = ['upper center','lower center','right','left','upper right','upper left','lower right','lower left',
                  'best','best']
       MAT_locs=['North','South','East','West','NorthEast', 'NorthWest', 'SouthEast', 'SouthWest','Best','none']
       Mat2py = dict(zip(MAT_locs,py_locs))
       location = legs.properties.Location
       plt.legend(leg_entries,loc=Mat2py[location])
   # plt.hold(False)
   plt.show()


plotFig('/home/anna/Downloads/10_1.6 volts_QuadRat-16_5-09-2017_RMG_13m-min_one_step.fig')