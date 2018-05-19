import tensorflow as tf


# Source https://stackoverflow.com/a/42166910
def block_diagonal(matrices, dtype=tf.float32):
    r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """

    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in
                matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(tensor=matrix,
                                 paddings=tf.concat([tf.zeros([tf.rank(matrix)
                                                               - 1, 2], dtype=tf.int32),
                                                     [(row_before_length, row_after_length)]],
                                                    axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows,
                                               blocked_cols)))
    return blocked
