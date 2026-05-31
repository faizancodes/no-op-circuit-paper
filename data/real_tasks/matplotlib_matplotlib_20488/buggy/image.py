                del A_scaled
                # un-scale the resampled data to approximately the
                # original range things that interpolated to above /
                # below the original min/max will still be above /
                # below, but possibly clipped in the case of higher order
                # interpolation + drastically changing data.
                A_resampled -= offset
                vrange -= offset
                if a_min != a_max:
                    A_resampled *= ((a_max - a_min) / frac)
                    vrange *= ((a_max - a_min) / frac)
                A_resampled += a_min
                vrange += a_min
                # if using NoNorm, cast back to the original datatype
                if isinstance(self.norm, mcolors.NoNorm):
                    A_resampled = A_resampled.astype(A.dtype)

                mask = (np.where(A.mask, np.float32(np.nan), np.float32(1))
                        if A.mask.shape == A.shape  # nontrivial mask
                        else np.ones_like(A, np.float32))
                # we always have to interpolate the mask to account for
                # non-affine transformations
                out_alpha = _resample(self, mask, out_shape, t, resample=True)
                # done with the mask now, delete from namespace to be sure!
                del mask
                # Agg updates out_alpha in place.  If the pixel has no image
                # data it will not be updated (and still be 0 as we initialized
                # it), if input data that would go into that output pixel than
                # it will be `nan`, if all the input data for a pixel is good
                # it will be 1, and if there is _some_ good data in that output
                # pixel it will be between [0, 1] (such as a rotated image).
                out_mask = np.isnan(out_alpha)
                out_alpha[out_mask] = 1
                # Apply the pixel-by-pixel alpha values if present
                alpha = self.get_alpha()
                if alpha is not None and np.ndim(alpha) > 0:
                    out_alpha *= _resample(self, alpha, out_shape,
                                           t, resample=True)
                # mask and run through the norm
                resampled_masked = np.ma.masked_array(A_resampled, out_mask)
                # we have re-set the vmin/vmax to account for small errors
                # that may have moved input values in/out of range
                s_vmin, s_vmax = vrange
                if isinstance(self.norm, mcolors.LogNorm):
                    if s_vmin < 0:
                        s_vmin = max(s_vmin, np.finfo(scaled_dtype).eps)
                with cbook._setattr_cm(self.norm,
                                       vmin=s_vmin,
                                       vmax=s_vmax,
                                       ):
                    output = self.norm(resampled_masked)
            else:
                if A.shape[2] == 3:
                    A = _rgb_to_rgba(A)
                alpha = self._get_scalar_alpha()
                output_alpha = _resample(  # resample alpha channel
                    self, A[..., 3], out_shape, t, alpha=alpha)
                output = _resample(  # resample rgb channels
                    self, _rgb_to_rgba(A[..., :3]), out_shape, t, alpha=alpha)
                output[..., 3] = output_alpha  # recombine rgb and alpha

            # at this point output is either a 2D array of normed data
            # (of int or float)
            # or an RGBA array of re-sampled input
            output = self.to_rgba(output, bytes=True, norm=False)
            # output is now a correctly sized RGBA array of uint8

            # Apply alpha *after* if the input was greyscale without a mask
            if A.ndim == 2:
                alpha = self._get_scalar_alpha()
                alpha_channel = output[:, :, 3]
                alpha_channel[:] = np.asarray(
                    np.asarray(alpha_channel, np.float32) * out_alpha * alpha,
                    np.uint8)

        else:
            if self._imcache is None:
                self._imcache = self.to_rgba(A, bytes=True, norm=(A.ndim == 2))
            output = self._imcache

            # Subset the input image to only the part that will be
            # displayed
            subset = TransformedBbox(clip_bbox, t0.inverted()).frozen()
            output = output[
                int(max(subset.ymin, 0)):
                int(min(subset.ymax + 1, output.shape[0])),
                int(max(subset.xmin, 0)):
                int(min(subset.xmax + 1, output.shape[1]))]
