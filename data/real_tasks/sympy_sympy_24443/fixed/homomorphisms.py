
def homomorphism(domain, codomain, gens, images=(), check=True):
    '''
    Create (if possible) a group homomorphism from the group ``domain``
    to the group ``codomain`` defined by the images of the domain's
    generators ``gens``. ``gens`` and ``images`` can be either lists or tuples
    of equal sizes. If ``gens`` is a proper subset of the group's generators,
    the unspecified generators will be mapped to the identity. If the
    images are not specified, a trivial homomorphism will be created.

    If the given images of the generators do not define a homomorphism,
    an exception is raised.

    If ``check`` is ``False``, do not check whether the given images actually
    define a homomorphism.

    '''
    if not isinstance(domain, (PermutationGroup, FpGroup, FreeGroup)):
        raise TypeError("The domain must be a group")
    if not isinstance(codomain, (PermutationGroup, FpGroup, FreeGroup)):
        raise TypeError("The codomain must be a group")

    generators = domain.generators
    if not all(g in generators for g in gens):
        raise ValueError("The supplied generators must be a subset of the domain's generators")
    if not all(g in codomain for g in images):
        raise ValueError("The images must be elements of the codomain")

    if images and len(images) != len(gens):
        raise ValueError("The number of images must be equal to the number of generators")

    gens = list(gens)
    images = list(images)

    images.extend([codomain.identity]*(len(generators)-len(images)))
    gens.extend([g for g in generators if g not in gens])
    images = dict(zip(gens,images))

    if check and not _check_homomorphism(domain, codomain, images):
        raise ValueError("The given images do not define a homomorphism")
    return GroupHomomorphism(domain, codomain, images)

def _check_homomorphism(domain, codomain, images):
    """
    Check that a given mapping of generators to images defines a homomorphism.

    Parameters
    ==========
    domain : PermutationGroup, FpGroup, FreeGroup
    codomain : PermutationGroup, FpGroup, FreeGroup
    images : dict
        The set of keys must be equal to domain.generators.
        The values must be elements of the codomain.

    """
    pres = domain if hasattr(domain, 'relators') else domain.presentation()
    rels = pres.relators
    gens = pres.generators
    symbols = [g.ext_rep[0] for g in gens]
    symbols_to_domain_generators = dict(zip(symbols, domain.generators))
    identity = codomain.identity

    def _image(r):
        w = identity
        for symbol, power in r.array_form:
            g = symbols_to_domain_generators[symbol]
            w *= images[g]**power
        return w

    for r in rels:
        if isinstance(codomain, FpGroup):
            s = codomain.equals(_image(r), identity)
            if s is None:
                # only try to make the rewriting system
                # confluent when it can't determine the
                # truth of equality otherwise
                success = codomain.make_confluent()
                s = codomain.equals(_image(r), identity)
                if s is None and not success:
                    raise RuntimeError("Can't determine if the images "
                        "define a homomorphism. Try increasing "
                        "the maximum number of rewriting rules "
                        "(group._rewriting_system.set_max(new_value); "
                        "the current value is stored in group._rewriting"
                        "_system.maxeqns)")
        else:
            s = _image(r).is_identity
        if not s:
            return False
    return True

def orbit_homomorphism(group, omega):
    '''
    Return the homomorphism induced by the action of the permutation
    group ``group`` on the set ``omega`` that is closed under the action.

    '''
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.named_groups import SymmetricGroup
    codomain = SymmetricGroup(len(omega))
    identity = codomain.identity
    omega = list(omega)
    images = {g: identity*Permutation([omega.index(o^g) for o in omega]) for g in group.generators}
    group._schreier_sims(base=omega)
    H = GroupHomomorphism(group, codomain, images)
    if len(group.basic_stabilizers) > len(omega):
        H._kernel = group.basic_stabilizers[len(omega)]
    else:
        H._kernel = PermutationGroup([group.identity])
    return H
