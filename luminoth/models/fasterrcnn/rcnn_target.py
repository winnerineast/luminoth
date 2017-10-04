import tensorflow as tf
import sonnet as snt
import numpy as np

from luminoth.utils.bbox_transform import encode, unmap
from luminoth.utils.bbox_transform_tf import encode as encode_tf
from luminoth.utils.bbox_overlap import bbox_overlap, bbox_overlap_tf


def assert_similar(expected_labels, actual_labels, expected_targets,
                   actual_targets, atol=1e-06):
    with tf.name_scope('assert_similar'):
        expected_sorted_top_k = tf.nn.top_k(
            expected_labels, tf.shape(expected_labels)[0]
        )
        actual_sorted_top_k = tf.nn.top_k(
            actual_labels, tf.shape(actual_labels)[0]
        )

        assert_sorted_labels_equals = tf.assert_equal(
            expected_labels, actual_labels,
            summarize=1e4
        )

        # assert_sorted_labels_equals = tf.assert_equal(
        #     expected_sorted_top_k.values, actual_sorted_top_k.values,
        #     summarize=1e4
        # )

        both_foreground = tf.logical_and(
            tf.greater(actual_labels, 0),
            tf.greater(expected_labels, 0)
        )

        both_foreground = tf.reshape(both_foreground, [-1])

        expected_targets = tf.boolean_mask(
            expected_targets, both_foreground
        )

        actual_targets = tf.boolean_mask(
            actual_targets, both_foreground
        )

        rdiff = tf.abs(
            expected_targets - actual_targets, 'diff'
        )
        atol = tf.convert_to_tensor(atol, name='rtol')
        assert_foreground_targets_close = tf.assert_less(
            rdiff,
            atol,
            data=(
                'Condition expected =~ actual did not hold element-wise:'
                'expected = ', expected_labels, 'actual = ', actual_labels,
                'rdiff = ', rdiff, 'atol = ', atol,
            ),
            summarize=1e4)

        return [assert_sorted_labels_equals, assert_foreground_targets_close]


class RCNNTarget(snt.AbstractModule):
    """Generate RCNN target tensors for both probabilities and bounding boxes.
    Targets for RCNN are based upon the results of the RPN, this can get tricky
    in the sense that RPN results might not be the best and it might not be
    possible to have the ideal amount of targets for all the available ground
    truth boxes.
    There are two types of targets, class targets and bounding box targets.
    Class targets are used both for background and foreground, while bounding
    box targets are only used for foreground (since it's not possible to create
    a bounding box of "background objects").
    A minibatch size determines how many targets are going to be generated and
    how many are going to be ignored. RCNNTarget is responsable for choosing
    which proposals and corresponding targets are included in the minibatch and
    which ones are completly ignored.
    TODO: Estimate performance degradation when running py_func.
    """
    def __init__(self, num_classes, config, seed=None, name='rcnn_proposal'):
        """
        Args:
            num_classes: Number of possible classes.
            config: Configuration object for RCNNTarget.
        """
        super(RCNNTarget, self).__init__(name=name)
        self._num_classes = num_classes
        # Ratio of foreground vs background for the minibatch.
        self._foreground_fraction = config.foreground_fraction
        self._minibatch_size = config.minibatch_size
        # IoU lower threshold with a ground truth box to be considered that
        # specific class.
        self._foreground_threshold = config.foreground_threshold
        # High and low treshold to be considered background.
        self._background_threshold_high = config.background_threshold_high
        self._background_threshold_low = config.background_threshold_low
        self._seed = seed

    def _build(self, proposals, gt_boxes):
        """
        RCNNTarget is implemented in numpy and attached to the computation
        graph using the `tf.py_func` operation.
        Ideally we should implement it as pure TensorFlow operation to avoid
        having to execute it in the CPU.
        Args:
            proposals: A Tensor with the RPN bounding boxes proposals.
                The shape of the Tensor is (num_proposals, 5), where the first
                of the 5 values for each proposal is the batch number.
            gt_boxes: A Tensor with the ground truth boxes for the image.
                The shape of the Tensor is (num_gt, 5), having the truth label
                as the last value for each box.
        Returns:
            proposals_label: Either a truth value of the proposals (a value
                between 0 and num_classes, with 0 being background), or -1 when
                the proposal is to be ignored in the minibatch.
                The shape of the Tensor is (num_proposals, 1).
            bbox_targets: A bounding box regression target for each of the
                proposals that have and greater than zero label. For every
                other proposal we return zeros.
                The shape of the Tensor is (num_proposals, 4).
        """
        (labels, bbox_targets) = tf.py_func(
            self._proposal_target_layer_np,
            [proposals, gt_boxes],
            [tf.float32, tf.float32],
            stateful=False,
            name='proposal_target_layer'
         )

        labels_t, bbox_targets_t = self._proposal_target_layer_tf(
            proposals, gt_boxes
        )

        asserts = assert_similar(
            labels_t, labels, bbox_targets_t, bbox_targets
        )

        with tf.control_dependencies(asserts):
            labels = tf.identity(labels)
            bbox_targets = tf.identity(bbox_targets)

        return labels, bbox_targets

    def _proposal_target_layer_np(self, proposals, gt_boxes):
        """
        Numpy implementation for _build.
        """
        np.random.seed(self._seed)

        # Remove batch id from proposals
        proposals = proposals[:, 1:]

        overlaps = bbox_overlap(proposals, gt_boxes)

        # overlaps returns (num_proposals, num_gt_boxes) with the IoU of
        # proposal P and ground truth box G in overlaps[P, G]

        # We are going to label each proposal based on the IoU with `gt_boxes`.
        # Start by filling the labels with -1, marking them as ignored.
        proposals_label = np.empty((proposals.shape[0], ), dtype=np.float32)
        proposals_label.fill(-1)

        # For each overlap there is three possible outcomes for labelling:
        #  if max(iou) < 0.1 then we ignore
        #  elif max(iou) <= 0.5 then we label background
        #  elif max(iou) > 0.5 then we label with the highest IoU in overlap
        max_overlaps = overlaps.max(axis=1)

        # Label background
        proposals_label[
            (max_overlaps >= self._background_threshold_low) &
            (max_overlaps < self._background_threshold_high)
        ] = 0

        # Filter proposal that have labels
        overlaps_with_label = max_overlaps >= self._foreground_threshold
        # Get label for proposal with labels
        overlaps_best_label = overlaps.argmax(axis=1)
        # Having the index of the gt bbox with the best label we need to get
        # the label for each gt box and sum it one because 0 is used for
        # background.
        # we only assign to proposals with `overlaps_with_label`.
        proposals_label[overlaps_with_label] = (
            gt_boxes[:, 4][overlaps_best_label] + 1
        )[overlaps_with_label]

        # Finally we get the closest proposal for each ground truth box and
        # mark it as positive.
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        proposals_label[gt_argmax_overlaps] = gt_boxes[:, 4] + 1

        # proposals_label now has [0, num_classes + 1] for proposals we are
        # going to use and -1 for the ones we should ignore.

        # Now we subsample labels and mark them as -1 in order to ignore them.
        # Our batch of N should be: F% foreground (label > 0)
        num_fg = int(self._foreground_fraction * self._minibatch_size)
        fg_inds = np.where(proposals_label >= 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False
            )
            # We disable them with their negatives just to be able to debug
            # down the road.
            proposals_label[disable_inds] = - proposals_label[disable_inds]

        if len(fg_inds) < num_fg:
            # When our foreground samples are not as much as we would like them
            # to be, we log a warning message.
            tf.logging.warning(
                'We\'ve got only {} foreground samples instead of {}.'.format(
                    len(fg_inds), num_fg
                )
            )

        # subsample negative labels
        num_bg = self._minibatch_size - np.sum(proposals_label >= 1)
        bg_inds = np.where(proposals_label == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False
            )
            proposals_label[disable_inds] = -1

        """
        Next step is to calculate the proper targets for the proposals labeled
        based on the values of the ground-truth boxes.
        We have to use only the proposals labeled >= 1, each matching with
        the proper gt_boxes
        """

        # Get the ids of the proposals that matter for bbox_target comparisson.
        proposal_with_target_idx = np.nonzero(proposals_label > 0)[0]

        # Get top gt_box for every proposal, top_gt_idx shape (1000,) with
        # values < gt_boxes.shape[0]
        top_gt_idx = overlaps.argmax(axis=1)

        # Get the corresponding ground truth box only for the proposals with
        # target.
        gt_boxes_ids = top_gt_idx[proposal_with_target_idx]

        # Get the values of the ground truth boxes. This is shaped
        # (num_proposals, 5) because we also have the label.
        proposals_gt_boxes = gt_boxes[gt_boxes_ids]

        # We create the same array but with the proposals
        proposals_with_target = proposals[proposal_with_target_idx]

        # We create our targets with bbox_transform
        bbox_targets = encode(
            proposals_with_target, proposals_gt_boxes
        )
        # TODO: We should normalize it in order for bbox_targets to have zero
        # mean and unit variance according to the paper.

        # We unmap targets to proposal_labels (containing the length of
        # proposals)
        bbox_targets = unmap(
            bbox_targets, proposals_label.shape[0], proposal_with_target_idx,
            fill=0
        )

        return proposals_label, bbox_targets

    def _proposal_target_layer_tf(self, proposals, gt_boxes):
        """
        Args:
            proposals: A Tensor with the RPN bounding boxes proposals.
                The shape of the Tensor is (num_proposals, 5), where the first
                of the 5 values for each proposal is the batch number.
            gt_boxes: A Tensor with the ground truth boxes for the image.
                The shape of the Tensor is (num_gt, 5), having the truth label
                as the last value for each box.
        Returns:
            proposals_label: Either a truth value of the proposals (a value
                between 0 and num_classes, with 0 being background), or -1 when
                the proposal is to be ignored in the minibatch.
                The shape of the Tensor is (num_proposals, 1).
            bbox_targets: A bounding box regression target for each of the
                proposals that have and greater than zero label. For every
                other proposal we return zeros.
                The shape of the Tensor is (num_proposals, 4).
        """
        # TODO: revise casts to int64 in tf.sparse_to_dense and tf.scatter_nd
        # calls.

        # Remove batch id from proposals.
        proposals = proposals[:, 1:]

        overlaps = bbox_overlap_tf(proposals, gt_boxes[:, :4])
        # overlaps now contains (num_proposals, num_gt_boxes) with the IoU of
        # proposal P and ground truth box G in overlaps[P, G]

        # We are going to label each proposal based on the IoU with
        # `gt_boxes`. Start by filling the labels with -1, marking them as
        # ignored.
        proposals_label_shape = tf.gather(tf.shape(proposals), [0])
        proposals_label = tf.fill(
            dims=proposals_label_shape,
            value=-1.
        )
        # For each overlap there is three possible outcomes for labelling:
        #  if max(iou) < config.background_threshold_low then we ignore.
        #  elif max(iou) <= config.background_threshold_high then we label
        #      background.
        #  elif max(iou) > config.foreground_threshold then we label with
        #      the highest IoU in overlap.
        #
        # max_overlaps gets, for each proposal, the index in which we can
        # find the gt_box with which it has the highest overlap.
        max_overlaps = tf.reduce_max(overlaps, axis=1)

        iou_is_high_enough_for_bg = tf.greater_equal(
            max_overlaps, self._background_threshold_low
        )
        iou_is_not_too_high_for_bg = tf.less(
            max_overlaps, self._background_threshold_high
        )
        bg_condition = tf.logical_and(
            iou_is_high_enough_for_bg, iou_is_not_too_high_for_bg
        )
        proposals_label = tf.where(
            condition=bg_condition,
            x=tf.zeros_like(proposals_label, dtype=tf.float32),
            y=proposals_label
        )

        # Get the index of the best gt_box for each proposal.
        overlaps_best_gt_idxs = tf.argmax(overlaps, axis=1)
        # Having the index of the gt bbox with the best label we need to get
        # the label for each gt box and sum it one because 0 is used for
        # background.
        best_fg_labels_for_proposals = tf.add(
            tf.gather(gt_boxes[:, 4], overlaps_best_gt_idxs),
            1.
        )
        iou_is_fg = tf.greater_equal(
            max_overlaps, self._foreground_threshold
        )
        best_proposals_idxs = tf.argmax(overlaps, axis=0)

        # Set the indices in best_proposals_idxs to True, and the rest to
        # false.
        # tf.sparse_to_dense is used because we know the set of indices which
        # we want to set to True, and we know the rest of the indices
        # should be set to False. That's exactly the use case of
        # tf.sparse_to_dense.
        is_best_box = tf.sparse_to_dense(
            sparse_indices=tf.reshape(best_proposals_idxs, [-1]),
            sparse_values=True, default_value=False,
            output_shape=tf.cast(proposals_label_shape, tf.int64),
            validate_indices=False
        )

        fg_condition = tf.logical_or(
            iou_is_fg, is_best_box
        )
        # We update proposals_label with the value in
        # best_fg_labels_for_proposals only when the box is foreground, or it
        # is the best proposal for a certain gt_box.
        proposals_label = tf.where(
            condition=fg_condition,
            x=best_fg_labels_for_proposals,
            y=proposals_label
        )

        # proposals_label now has a value in [0, num_classes + 1] for
        # proposals we are going to use and -1 for the ones we should ignore.
        # But we still need to make sure we don't have a number of proposals
        # higher than minibatch_size * foreground_fraction.
        max_fg = int(self._foreground_fraction * self._minibatch_size)
        fg_inds = tf.where(
            condition=fg_condition
        )

        def disable_some_fgs():
            # We want to delete a randomly-selected subset of fg_inds of
            # size `fg_inds.shape[0] - max_fg`.
            # We shuffle along the dimension 0 and then we get the first
            # num_fg_inds - max_fg indices and we disable them.
            shuffled_inds = tf.random_shuffle(fg_inds, seed=self._seed)
            disable_place = (tf.shape(fg_inds)[0] - max_fg)
            # This function should never run if num_fg_inds <= max_fg, so we
            # add an assertion to catch the wrong behaviour if it happens.
            integrity_assertion = tf.assert_positive(
                disable_place,
                message="disable_place in disable_some_fgs is negative."
            )
            with tf.control_dependencies([integrity_assertion]):
                disable_inds = shuffled_inds[:disable_place]
            is_disabled = tf.sparse_to_dense(
                sparse_indices=disable_inds,
                sparse_values=True, default_value=False,
                output_shape=tf.cast(proposals_label_shape, tf.int64),
                # We are shuffling the indices, so they may not be ordered.
                validate_indices=False
            )
            return tf.where(
                condition=is_disabled,
                # We set it to -label for debugging purposes.
                x=tf.negative(proposals_label),
                y=proposals_label
            )
        # Disable some fgs if we have too many foregrounds.
        proposals_label = tf.cond(
            tf.greater(tf.shape(fg_inds)[0], max_fg),
            true_fn=disable_some_fgs,
            false_fn=lambda: proposals_label
        )

        total_fg_in_batch = tf.shape(
            tf.where(
                condition=tf.greater(proposals_label, 0)
            )
        )[0]

        # Now we want to do the same for backgrounds.
        # We calculate up to how many backgrounds we desire based on the
        # final number of foregrounds and the total desired batch size.
        max_bg = self._minibatch_size - total_fg_in_batch

        # We can't use bg_condition because some of the proposals that satisfy
        # the IoU conditions to be background may have been labeled as
        # foreground due to them being the best proposal for a certain gt_box.
        bg_mask = tf.equal(proposals_label, 0)
        bg_inds = tf.where(
            condition=bg_mask,
        )

        def disable_some_bgs():
            # Mutatis mutandis, all comments from disable_some_fgs apply.
            shuffled_inds = tf.random_shuffle(bg_inds, seed=self._seed)
            disable_place = (tf.shape(bg_inds)[0] - max_bg)
            integrity_assertion = tf.assert_non_negative(
                disable_place,
                message="disable_place in disable_some_bgs is negative."
            )
            with tf.control_dependencies([integrity_assertion]):
                disable_inds = shuffled_inds[:disable_place]
            is_disabled = tf.sparse_to_dense(
                sparse_indices=disable_inds,
                sparse_values=True, default_value=False,
                output_shape=tf.cast(proposals_label_shape, tf.int64),
                validate_indices=False
            )
            return tf.where(
                condition=is_disabled,
                x=tf.fill(
                    dims=proposals_label_shape,
                    value=-1.
                ),
                y=proposals_label
            )

        proposals_label = tf.cond(
            tf.greater_equal(tf.shape(bg_inds)[0], max_bg),
            true_fn=disable_some_bgs,
            false_fn=lambda: proposals_label
        )

        """
        Next step is to calculate the proper targets for the proposals labeled
        based on the values of the ground-truth boxes.
        We have to use only the proposals labeled >= 1, each matching with
        the proper gt_boxes
        """

        # Get the ids of the proposals that matter for bbox_target comparisson.
        is_proposal_with_target = tf.greater(
            proposals_label, 0
        )
        proposals_with_target_idx = tf.where(
            condition=is_proposal_with_target
        )
        # Get the corresponding ground truth box only for the proposals with
        # target.
        gt_boxes_idxs = tf.gather(
            overlaps_best_gt_idxs,
            proposals_with_target_idx
        )
        # Get the values of the ground truth boxes.
        proposals_gt_boxes = tf.gather_nd(
            gt_boxes[:, :4], gt_boxes_idxs
        )
        # We create the same array but with the proposals
        proposals_with_target = tf.gather_nd(
            proposals,
            proposals_with_target_idx
        )
        # We create our targets with bbox_transform
        bbox_targets_nonzero = encode_tf(
            proposals_with_target,
            proposals_gt_boxes,
        )
        # TODO: We should normalize it in order for bbox_targets to have zero
        # mean and unit variance according to the paper.

        # We unmap targets to proposal_labels (containing the length of
        # proposals)
        bbox_targets = tf.scatter_nd(
            indices=proposals_with_target_idx,
            updates=bbox_targets_nonzero,
            shape=tf.cast(tf.shape(proposals), tf.int64)
        )

        proposals_label = proposals_label
        bbox_targets = bbox_targets

        return proposals_label, bbox_targets
