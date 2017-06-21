import importlib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import datasets


def gumbel_softmax(x, temperature):
    with tf.variable_scope('concrete'):
        # sample from Gumbel distribution via `g = -log(-log(u))` where u ~ Uniform(0, 1)
        g = -tf.log(-tf.log(tf.random_uniform(tf.shape(x))))
        logits = (x + g) / temperature
        # softmax with temperature
    return logits


class FastSaver(tf.train.Saver):
    # HACK disable saving metagraphs
    def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True, write_state=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename, meta_graph_suffix, False, write_state)


if __name__ == '__main__':
    import argparse, os


    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('-l', '--log-dir', default='/tmp/mnist')
    parser.add_argument('-n', '--max-iter', type=int, default=1000)
    parser.add_argument('-m', '--model', default='cnn')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-t', '--temperature', type=float, default=1., help='softmax temperature for concrete relaxation')
    parser.add_argument('-a', '--baseline', type=float, default=1. / 10., help='constant baseline for policy gradient')
    parser.add_argument('-o', '--objective', choices=['relaxed', 'supervised', 'sample'], default='supervised')
    parser.add_argument('--summary-interval', type=int, default=32)
    parser.add_argument('--test-interval', type=int, default=128)
    # TODO add a train/val//test mode
    # TODO add random seed and determinism

    args, extra = parser.parse_known_args()

    # import model parser and builder
    model = importlib.import_module('models.%s' % args.model)
    model_parser = model.build_parser()
    model_args, extra = model_parser.parse_known_args(extra)
    build_model = lambda input: model.build_model(input, **vars(model_args))

    # optimizer parameters
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop', 'momentum'], default='adam')
    opt_args = opt_parser.parse_args(extra)
    # TODO add more optimizer-specific arguments

    # training
    image_ph = tf.placeholder('float', [None, 28, 28, 1], 'images')
    batch_size = tf.cast(tf.shape(image_ph)[0], 'float')

    with tf.variable_scope('model'):
        logits = build_model(image_ph)
        if args.objective == 'relaxed':
            logits = gumbel_softmax(logits, args.temperature)

    variables = tf.trainable_variables()
    print '* trainable variables:'
    for v in variables:
        print v

    # objective
    label_ph = tf.placeholder('int64', [None], 'labels')
    if args.objective == 'sample':
        # stochastic, policy gradient
        yhat = tf.squeeze(tf.multinomial(logits, 1), -1)
        baseline = args.baseline
        r = tf.expand_dims(tf.cast(tf.equal(yhat, label_ph), 'float'), -1) - baseline
        # to maximize
        loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * tf.one_hot(yhat, 10) * r, name='policy_gradient_loss') / batch_size
    else:
        # to minimize cross entropy
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label_ph, logits=logits)

    # training
    global_step = tf.contrib.framework.get_or_create_global_step()
    opt_dict = {
        'adam': tf.train.AdamOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer,
        'momentum': tf.train.MomentumOptimizer,
    }
    optimizer = opt_dict[opt_args.optimizer](
        learning_rate=args.lr,
    )
    grads = tf.gradients(loss * batch_size, variables)
    update_op = optimizer.apply_gradients(zip(grads, variables), global_step=global_step)

    # summaries
    tf.summary.scalar('train/variable_norm', tf.global_norm(variables))
    tf.summary.scalar('train/gradient_norm', tf.global_norm(grads))
    tf.summary.scalar('train/loss', loss)
    tf.summary.scalar('train/learning_rate', args.lr)
    if args.objective == 'relaxed':
        tf.summary.scalar('train/temperature', args.temperature)
    for g, v in zip(grads, variables):
        tf.summary.histogram('gradients/%s' % v.name, g / batch_size)
    summary_op = tf.summary.merge_all()

    # test
    test_image_ph = tf.placeholder('float', [None, 28, 28, 1], 'test_images')
    test_batch_size = tf.cast(tf.shape(test_image_ph)[0], 'float')
    with tf.variable_scope('model', reuse=True):
        test_logits = build_model(test_image_ph)

    test_label_ph = tf.placeholder('int64', [None], 'test_labels')
    test_loss = tf.losses.sparse_softmax_cross_entropy(labels=test_label_ph, logits=test_logits)

    correct_prediction = tf.equal(tf.argmax(test_logits, 1), test_label_ph)
    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # to placeholders to accumulate
    test_loss_ph = tf.placeholder('float', [], 'test_loss_ph')
    test_accuracy_ph = tf.placeholder('float', [], 'test_accuracy_ph')
    test_summary_op = tf.summary.merge([
        tf.summary.scalar('test/loss', test_loss_ph),
        tf.summary.scalar('test/accuracy', test_accuracy_ph),
    ])

    # load dataset
    # TODO switch to keras datasets to support a few more small datasets?
    mnist = datasets.load_dataset('mnist')
    test_x, test_y = mnist.test.images, mnist.test.labels
    test_x = test_x.reshape(-1, 28, 28, 1)
    n_test = len(test_x)

    def do_test():
        c_test_loss = 0.
        c_test_acc = 0.
        for i in xrange(int(np.ceil(n_test / args.batch_size))):
            start = i * args.batch_size
            end = min(n_test, start + args.batch_size)
            n_batch = end - start
            test_feed = {
                test_image_ph: test_x[start:end],
                test_label_ph: test_y[start:end],
            }
            test_batch_loss, test_batch_acc = sess.run([test_loss, test_accuracy], test_feed)
            c_test_loss += test_batch_loss * n_batch
            c_test_acc += test_batch_acc * n_batch

        return c_test_loss / n_test, c_test_acc / n_test

    writer = tf.summary.FileWriter(args.log_dir, flush_secs=10)
    saver = FastSaver(keep_checkpoint_every_n_hours=1, max_to_keep=2)
    sv = tf.train.Supervisor(
        saver=saver,
        ready_op=tf.report_uninitialized_variables(),
        logdir=args.log_dir,
        summary_op=None,
        save_model_secs=30,
    )

    config = tf.ConfigProto(
        gpu_options={
            'allow_growth': True,
        }
    )

    with sv.managed_session('', config=config) as sess, sess.as_default():
        gs = global_step.eval()
        for _ in xrange(args.max_iter):
            x, y = mnist.train.next_batch(args.batch_size)
            x = x.reshape(-1, 28, 28, 1)
            feed = {
                image_ph: x,
                label_ph: y,
            }

            if gs % args.test_interval == 0:
                # evaluate on the test split
                test_loss_val, test_acc_val = do_test()
                print '%i test loss %g test accuracy %g' % (gs, test_loss_val, test_acc_val)
                test_summary_val = test_summary_op.eval(feed_dict={
                    test_loss_ph: test_loss_val,
                    test_accuracy_ph: test_acc_val,
                })
                writer.add_summary(test_summary_val, global_step=gs)

            if gs % args.summary_interval == 0:
                summary_val, _, gs = sess.run([summary_op, update_op, global_step], feed_dict=feed)
                writer.add_summary(summary_val, global_step=gs)
                # save model
                saver.save(sess, os.path.join(args.log_dir, 'model'), gs)
            else:
                _, gs = sess.run([update_op, global_step], feed_dict=feed)

        # last test
        test_loss_val, test_acc_val = do_test()
        print '%i test loss %g test accuracy %g' % (gs, test_loss_val, test_acc_val)
        # save the last model
        saver.save(sess, os.path.join(args.log_dir, 'model'), gs)
    writer.close()
