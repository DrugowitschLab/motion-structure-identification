def draw_structure(ax, bbox, structure, zorder=0):
    """
    visualizes a motion structure inside a given bbox on a given axes
    :param ax: where the structures will be drawn
    :type ax: matplotlib.axes.Axes
    :param bbox: (x, y, w, h) bounding box inside which the structure will be drawn
    :type bbox: (float, float, float, float)
    :param structure: the structure to be drawn
    :type structure: str
    :param zorder: drawing order of the patches composing the structure
    :type zorder: int
    """
    from matplotlib.patches import Circle, ConnectionPatch
    x, y, w, h = bbox
    r = w / 10
    centers = [[(x + dx, y + dy) for (dx, dy) in layer] for layer in nodes[structure](r, h)]
    lines = edges[structure](centers)
    for layer in centers[:-1]:
        for c in layer:
            ax.add_patch(Circle(c, r, zorder=zorder))
    for c1, c2 in lines:
        ax.add_patch(ConnectionPatch(c1, c2, 'data', 'data', zorder=zorder-1))


nodes = {
    'IND': lambda r, h:
        [[(r, r), (r * 5, r), (r * 9, r)],
         [(r, h), (r * 5, h), (r * 9, h)]],
    'GLO': lambda r, h:
        [[(r, r), (r * 5, r), (r * 9, r)],
         [(r * 5, r + h / 2)],
         [(r * 5, h)]],
    'CLU': lambda r, h:
        [[(r, r), (r * 5, r), (r * 9, r)],
         [(r * 3, r + h / 2)],
         [(r * 3, h), (r * 9, h)]],
    'SDH': lambda r, h:
        [[(r, r), (r * 5, r), (r * 9, r)],
         [(r * 3, r + h / 3)],
         [(r * 6, r + h * 2 / 3)],
         [(r * 6, h)]],
}

edges = {
    'IND': lambda c:
        [(c[0][0], c[1][0]), (c[0][1], c[1][1]), (c[0][2], c[1][2])],
    'GLO': lambda c:
        [(c[0][0], c[1][0]), (c[0][1], c[1][0]), (c[0][2], c[1][0]),
         (c[1][0], c[2][0])],
    'CLU': lambda c:
        [(c[0][0], c[1][0]), (c[0][1], c[1][0]),
         (c[1][0], c[2][0]), (c[0][2], c[2][1])],
    'SDH': lambda c:
        [(c[0][0], c[1][0]), (c[0][1], c[1][0]),
         (c[1][0], c[2][0]), (c[0][2], c[2][0]),
         (c[2][0], c[3][0])],
}

if __name__ == '__main__':
    # unit test
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.text(0.15, -0.05, 'IND', ha='center', va='top')
    draw_structure(ax, (0, 0, 0.3, 0.5), 'IND')
    ax.text(0.55, -0.05, 'GLO', ha='center', va='top')
    draw_structure(ax, (0.4, 0, 0.3, 0.5), 'GLO')
    ax.text(0.95, -0.05, 'CLU', ha='center', va='top')
    draw_structure(ax, (0.8, 0, 0.3, 0.5), 'CLU')
    ax.text(1.35, -0.05, 'SDH', ha='center', va='top')
    draw_structure(ax, (1.2, 0, 0.3, 0.5), 'SDH')
    ax.axis('off')
    ax.axis('equal')
    ax.axis([0, 1.5, 0, 0.5])
    plt.show()
