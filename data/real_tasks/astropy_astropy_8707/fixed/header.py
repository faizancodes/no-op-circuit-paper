        View the comments associated with each keyword, if any.

        For example, to see the comment on the NAXIS keyword:

            >>> header.comments['NAXIS']
            number of data axes

        Comments can also be updated through this interface:

            >>> header.comments['NAXIS'] = 'Number of data axes'

        """

        return _HeaderComments(self)

    @property
    def _modified(self):
        """
        Whether or not the header has been modified; this is a property so that
        it can also check each card for modifications--cards may have been
        modified directly without the header containing it otherwise knowing.
        """

        modified_cards = any(c._modified for c in self._cards)
        if modified_cards:
            # If any cards were modified then by definition the header was
            # modified
            self.__dict__['_modified'] = True

        return self.__dict__['_modified']

    @_modified.setter
    def _modified(self, val):
        self.__dict__['_modified'] = val

    @classmethod
    def fromstring(cls, data, sep=''):
        """
        Creates an HDU header from a byte string containing the entire header
        data.

        Parameters
        ----------
        data : str or bytes
           String or bytes containing the entire header.  In the case of bytes
           they will be decoded using latin-1 (only plain ASCII characters are
           allowed in FITS headers but latin-1 allows us to retain any invalid
           bytes that might appear in malformatted FITS files).

        sep : str, optional
            The string separating cards from each other, such as a newline.  By
            default there is no card separator (as is the case in a raw FITS
            file).  In general this is only used in cases where a header was
            printed as text (e.g. with newlines after each card) and you want
            to create a new `Header` from it by copy/pasting.

        Examples
        --------

        >>> from astropy.io.fits import Header
        >>> hdr = Header({'SIMPLE': True})
        >>> Header.fromstring(hdr.tostring()) == hdr
        True

        If you want to create a `Header` from printed text it's not necessary
        to have the exact binary structure as it would appear in a FITS file,
        with the full 80 byte card length.  Rather, each "card" can end in a
        newline and does not have to be padded out to a full card length as
        long as it "looks like" a FITS header:

        >>> hdr = Header.fromstring(\"\"\"\\
        ... SIMPLE  =                    T / conforms to FITS standard
        ... BITPIX  =                    8 / array data type
        ... NAXIS   =                    0 / number of array dimensions
        ... EXTEND  =                    T
        ... \"\"\", sep='\\n')
        >>> hdr['SIMPLE']
        True
        >>> hdr['BITPIX']
        8
        >>> len(hdr)
        4

        Returns
        -------
        header
            A new `Header` instance.
        """

        cards = []

        # If the card separator contains characters that may validly appear in
        # a card, the only way to unambiguously distinguish between cards is to
        # require that they be Card.length long.  However, if the separator
        # contains non-valid characters (namely \n) the cards may be split
        # immediately at the separator
        require_full_cardlength = set(sep).issubset(VALID_HEADER_CHARS)

        # Split the header into individual cards
        idx = 0
        image = []

        while idx < len(data):
            if require_full_cardlength:
                end_idx = idx + Card.length
            else:
                try:
                    end_idx = data.index(sep, idx)
                except ValueError:
                    end_idx = len(data)

            next_image = data[idx:end_idx]
            idx = end_idx + len(sep)

            if image:
                if next_image[:8] == 'CONTINUE':
                    image.append(next_image)
                    continue
                cards.append(Card.fromstring(''.join(image)))

            if require_full_cardlength:
                if next_image == END_CARD:
                    image = []
                    break
            else:
