import matplotlib as mpl

name = "Bill's Figure Settings for Matplotlib"

cm2inch = lambda x: 0.393700787 * x


#for plos
fig_width = dict(
    one_col = cm2inch(7.),
    two_col = cm2inch(16.)
    )

font_panellabel = dict(fontweight='bold', fontsize=12, ha='left')

config = {
    'axes' : dict(labelsize=8, titlesize=8, linewidth=0.5),
    #'figure' : dict(dpi=200, figsize=[fig_width['one_col'], 0.75*fig_width['one_col']], facecolor='white'),
    'figure' : dict(dpi=114., figsize=[6., 6./1.60], facecolor='white'),
    'figure.subplot' : dict(left=0.10, bottom=0.12, right=0.97, top=0.97),
    'font' : {'family' : 'sans-serif', 'size' : 8, 'weight' : 'normal',
              'sans-serif' : ['Arial', 'LiberationSans-Regular', 'FreeSans']},
    'image' : dict(cmap='RdBu_r' , interpolation='nearest'),
    'legend' : dict(fontsize=8, borderaxespad=0.5, borderpad=0.5),
    'lines' : dict(linewidth=0.5),
    'xtick' : dict(labelsize=8),
    'xtick.major' : dict(size=1.5, pad=2, width=0.5),
    'ytick' : dict(labelsize=8),
    'ytick.major' : dict(size=1.5, pad=2, width=0.5),
    'savefig' : dict(dpi=300)
    }

print ("\n\t * * * Importing '%s' * * *\n" % name)

for key,val in config.items():
    s = ""
    for k,v in val.items():
        s += k + "=%s, " % str(v)
    print ("  > Set '%s' to %s" % (key, s[:-2]) )
    mpl.rc(key, **val)

print ('\n\n')
