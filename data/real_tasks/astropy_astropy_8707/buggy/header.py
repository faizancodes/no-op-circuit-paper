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
        data : str
           String containing the entire header.

        sep : str, optional
            The string separating cards from each other, such as a newline.  By
            default there is no card separator (as is the case in a raw FITS
            file).

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
