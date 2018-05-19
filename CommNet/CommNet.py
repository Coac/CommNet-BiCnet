import tensorflow as tf

NUM_AGENTS = 5

VECTOR_OBS_LEN = 10
OUTPUT = 2


def comm_step(name, H):
    with tf.variable_scope(name):
        w_H = tf.get_variable(name='w_H', shape=VECTOR_OBS_LEN, initializer=tf.constant_initializer(10))
        w_C = tf.get_variable(name='w_C', shape=VECTOR_OBS_LEN, initializer=tf.constant_initializer(1))

        normalized_w_C = tf.divide(w_C, NUM_AGENTS - 1)

        w_C_matrix = tf.reshape(tf.tile(normalized_w_C, [NUM_AGENTS]), (1, NUM_AGENTS * VECTOR_OBS_LEN))
        w_C_matrix = tf.tile(w_C_matrix, [NUM_AGENTS, 1])

        w_H_minus_w_C_matrix = block_diagonal(
            [tf.reshape(tf.subtract(w_H, normalized_w_C), (1, VECTOR_OBS_LEN)) for j in range(NUM_AGENTS)])
        T = tf.add(w_C_matrix, w_H_minus_w_C_matrix)

        H_repeated = tf.tile(H, [1, NUM_AGENTS])

        return tf.tanh(tf.reshape(tf.reduce_sum(H_repeated * T, axis=0), H.shape))


def output_layer(H):
    with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='w', shape=(VECTOR_OBS_LEN, OUTPUT), initializer=tf.constant_initializer(10))

        outputs = []
        for j in range(NUM_AGENTS):
            h = tf.slice(H, [j, 0], [1, VECTOR_OBS_LEN])

            outputs.append(tf.tanh(tf.matmul(h, w))[0])
        return outputs


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
    tf.set_random_seed(42)

    H0 = tf.random_uniform(shape=(NUM_AGENTS, VECTOR_OBS_LEN), dtype="float32")
    H1 = comm_step("comm_step1", H0)
    H2 = comm_step("comm_step2", H1)
    out = output_layer(H2)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("summaries", sess.graph)

        sess.run(tf.global_variables_initializer())

        res = sess.run(out)
        for i in res:
            print ('-----')
            print(i)

        # print(sess.run(output_layer(H).shape))

        writer.close()
