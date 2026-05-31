def _correct_predecessor(reachability_plot, predecessor_plot, ordering, s, e):
    """Correct for predecessors.

    Applies Algorithm 2 of [1]_.

    Input parameters are ordered by the computer OPTICS ordering.

    .. [1] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.
    """
    while s < e:
        if reachability_plot[s] > reachability_plot[e]:
            return s, e
        p_e = ordering[predecessor_plot[e]]
        for i in range(s, e):
            if p_e == ordering[i]:
                return s, e
        e -= 1
    return None, None


def _xi_cluster(reachability_plot, predecessor_plot, ordering, xi, min_samples,
                min_cluster_size, predecessor_correction):
    """Automatically extract clusters according to the Xi-steep method.

    This is rouphly an implementation of Figure 19 of the OPTICS paper.

    Parameters
    ----------
    reachability_plot : array, shape (n_samples)
        The reachability plot, i.e. reachability ordered according to
        the calculated ordering, all computed by OPTICS.

    predecessor_plot : array, shape (n_samples)
        Predecessors ordered according to the calculated ordering.

    xi : float, between 0 and 1
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.

    min_samples : int > 1
        The same as the min_samples given to OPTICS. Up and down steep regions
        can't have more then ``min_samples`` consecutive non-steep points.

    min_cluster_size : int > 1
        Minimum number of samples in an OPTICS cluster.

    predecessor_correction : bool
        Correct clusters based on the calculated predecessors.

    Returns
    -------
    clusters : array, shape (n_clusters, 2)
        The list of clusters in the form of [start, end] in each row, with all
        indices inclusive. The clusters are ordered in a way that larger
        clusters encompassing smaller clusters come after those smaller
        clusters.
    """

    # Our implementation adds an inf to the end of reachability plot
    # this helps to find potential clusters at the end of the
    # reachability plot even if there's no upward region at the end of it.
    reachability_plot = np.hstack((reachability_plot, np.inf))

    xi_complement = 1 - xi
    sdas = []  # steep down areas, introduced in section 4.3.2 of the paper
    clusters = []
    index = 0
    mib = 0.  # maximum in between, section 4.3.2

    # Our implementation corrects a mistake in the original
    # paper, i.e., in Definition 9 steep downward point,
    # r(p) * (1 - x1) <= r(p + 1) should be
    # r(p) * (1 - x1) >= r(p + 1)
    with np.errstate(invalid='ignore'):
        ratio = reachability_plot[:-1] / reachability_plot[1:]
        steep_upward = ratio <= xi_complement
        steep_downward = ratio >= 1 / xi_complement
        downward = ratio > 1
        upward = ratio < 1

    # the following loop is is almost exactly as Figure 19 of the paper.
    # it jumps over the areas which are not either steep down or up areas
    for steep_index in iter(np.flatnonzero(steep_upward | steep_downward)):
        # just continue if steep_index has been a part of a discovered xward
        # area.
        if steep_index < index:
            continue
