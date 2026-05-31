        connected_components += 1
        pointer = min(links.keys())
        previous_pointer = pointer
        # the inner loop erases the visited indices from `links`, and it adds
        # all free indices to `prepend_indices` list, virtual indices are
        # ignored.
        prepend_indices = []
        while True:
            if pointer in links:
                next_ones = links.pop(pointer)
            else:
                break

            if previous_pointer in next_ones:
                if len(next_ones) > 1:
                    next_ones.remove(previous_pointer)

            previous_pointer = pointer

            if next_ones:
                pointer = next_ones[0]

            if pointer >= first_dum_pos and free_pos[pointer] is not None:
                prepend_indices.insert(0, free_pos[pointer])
        # if `prepend_indices` is void, it means there are no free indices
        # in the loop (and it can be shown that there must be a virtual index),
        # loops of virtual indices only contribute by a factor of two:
        if len(prepend_indices) == 0:
            resulting_coeff *= 2
        # otherwise, add the free indices in `prepend_indices` to
        # the `resulting_indices`:
        else:
            expr1 = prepend_indices
            expr2 = list(reversed(prepend_indices))
            resulting_indices = [expri + ri for ri in resulting_indices for expri in (expr1, expr2)]

    # sign correction, as described in Kahane's paper:
    resulting_coeff *= -1 if (number_of_contractions - connected_components + 1) % 2 else 1
    # power of two factor, as described in Kahane's paper:
    resulting_coeff *= 2**(number_of_contractions)

    # If `first_dum_pos` is not zero, it means that there are trailing free gamma
    # matrices in front of `expression`, so multiply by them:
    resulting_indices = list( free_pos[0:first_dum_pos] + ri for ri in resulting_indices )

    resulting_expr = S.Zero
    for i in resulting_indices:
        temp_expr = S.One
        for j in i:
            temp_expr *= GammaMatrix(j)
        resulting_expr += temp_expr

    t = resulting_coeff * resulting_expr
    t1 = None
    if isinstance(t, TensAdd):
        t1 = t.args[0]
    elif isinstance(t, TensMul):
        t1 = t
    if t1:
        pass
    else:
        t = eye(4)*t
    return t
