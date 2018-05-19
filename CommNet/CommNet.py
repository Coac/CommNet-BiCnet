import tensorflow as tf

NUM_AGENTS = 5
SHAPE = (10, 10)


def module(hidden_state, communication):
    with tf.variable_scope('module', reuse=tf.AUTO_REUSE):
        w_H = tf.get_variable(name='w_H', shape=hidden_state.shape, initializer=tf.random_normal_initializer(0, 0.2))
        w_C = tf.get_variable(name='w_C', shape=communication.shape, initializer=tf.random_normal_initializer(0, 0.2))

    return tf.tanh(tf.add(tf.matmul(w_H, hidden_state), (tf.matmul(w_C, communication))))


def comm_step(name, H):
    H_shape = H.shape
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # w_H = tf.get_variable(name='w_H', shape=SHAPE, initializer=tf.random_normal_initializer(0, 0.2))
        # w_C = tf.get_variable(name='w_C', shape=SHAPE, initializer=tf.random_normal_initializer(0, 0.2))

        w_H = tf.get_variable(name='w_H', shape=SHAPE, initializer=tf.constant_initializer(0.99))
        w_C = tf.get_variable(name='w_C', shape=SHAPE, initializer=tf.constant_initializer(1))

        normalized_w_C = tf.divide(w_C, NUM_AGENTS - 1)
        w_C_matrix = tf.tile(normalized_w_C, [NUM_AGENTS, NUM_AGENTS])
        w_H_minus_w_C_matrix = block_diagonal([tf.subtract(w_H, normalized_w_C) for i in range(NUM_AGENTS)])

        T = tf.add(w_C_matrix, w_H_minus_w_C_matrix)

        print(H_shape)
        return tf.matmul(T, tf.reshape(H, (SHAPE[1] * NUM_AGENTS, SHAPE[0])))


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


if __name__ == '__main__':
    sess = tf.Session()
    H = tf.random_uniform(shape=(NUM_AGENTS, SHAPE[0], SHAPE[1]), dtype="float32")

    out = comm_step("comm_step1", H)

    sess.run(tf.global_variables_initializer())

    print(sess.run(out))
    print(sess.run(out).shape)
