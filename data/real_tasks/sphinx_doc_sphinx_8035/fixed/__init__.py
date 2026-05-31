
            # if the member __doc__ is the same as self's __doc__, it's just
            # inherited and therefore not the member's doc
            cls = self.get_attr(member, '__class__', None)
            if cls:
                cls_doc = self.get_attr(cls, '__doc__', None)
                if cls_doc == doc:
                    doc = None
            has_doc = bool(doc)

            metadata = extract_metadata(doc)
            if 'private' in metadata:
                # consider a member private if docstring has "private" metadata
                isprivate = True
            elif 'public' in metadata:
                # consider a member public if docstring has "public" metadata
                isprivate = False
            else:
                isprivate = membername.startswith('_')

            keep = False
            if safe_getattr(member, '__sphinx_mock__', False):
                # mocked module or object
                pass
            elif (self.options.exclude_members not in (None, ALL) and
                  membername in self.options.exclude_members):
                # remove members given by exclude-members
                keep = False
            elif want_all and membername.startswith('__') and \
                    membername.endswith('__') and len(membername) > 4:
                # special __methods__
                if self.options.special_members is ALL:
                    if membername == '__doc__':
                        keep = False
                    elif is_filtered_inherited_member(membername):
                        keep = False
                    else:
                        keep = has_doc or self.options.undoc_members
                elif self.options.special_members:
                    if membername in self.options.special_members:
                        keep = has_doc or self.options.undoc_members
            elif (namespace, membername) in attr_docs:
                if want_all and isprivate:
                    if self.options.private_members is None:
                        keep = False
                    elif self.options.private_members is ALL:
                        keep = True
                    else:
                        keep = membername in self.options.private_members
                else:
                    # keep documented attributes
                    keep = True
                isattr = True
            elif want_all and isprivate:
                if has_doc or self.options.undoc_members:
                    if self.options.private_members is None:
                        keep = False
                    elif self.options.private_members is ALL:
                        keep = True
                    elif is_filtered_inherited_member(membername):
                        keep = False
                    else:
                        keep = membername in self.options.private_members
                else:
                    keep = False
            else:
                if self.options.members is ALL and is_filtered_inherited_member(membername):
                    keep = False
                else:
                    # ignore undocumented members if :undoc-members: is not given
                    keep = has_doc or self.options.undoc_members

            # give the user a chance to decide whether this member
            # should be skipped
            if self.env.app:
                # let extensions preprocess docstrings
                try:
                    skip_user = self.env.app.emit_firstresult(
                        'autodoc-skip-member', self.objtype, membername, member,
                        not keep, self.options)
                    if skip_user is not None:
                        keep = not skip_user
                except Exception as exc:
                    logger.warning(__('autodoc: failed to determine %r to be documented, '
                                      'the following exception was raised:\n%s'),
                                   member, exc, type='autodoc')
                    keep = False

            if keep:
                ret.append((membername, member, isattr))

        return ret

    def document_members(self, all_members: bool = False) -> None:
        """Generate reST for member documentation.

        If *all_members* is True, do all members, else those given by
        *self.options.members*.
        """
        # set current namespace for finding members
        self.env.temp_data['autodoc:module'] = self.modname
        if self.objpath:
            self.env.temp_data['autodoc:class'] = self.objpath[0]

        want_all = all_members or self.options.inherited_members or \
            self.options.members is ALL
        # find out which members are documentable
        members_check_module, members = self.get_object_members(want_all)
