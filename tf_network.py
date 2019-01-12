import tensorflow as tf
import numpy as np
from random import shuffle
import pickle
import os
import datetime
import time
import glob
import pandas as pd


# use from_generator instead of current implementation.
def as_matrix(config):
    return [[k, str(w)] for k, w in config.items()]

def date_convert(date):
    ori_date = date
    year = int(date[:date.find("-")])
    date = date[date.find("-")+1:]
    month = int(date[:date.find("-")])
    date = date[date.find("-")+1:]
    day = int(date)
    try:
        to_return = datetime.datetime(year,month,day).weekday()
    except Exception as e:
        print(ori_date,e,year,month,day)
        exit()
    return to_return

def unbounded_int(i,medium):
    if i == 0:
        return 0
    elif i <= medium:
        return 1
    else:
        return 2

def make_onehot(i, cats):
    to_fill = np.zeros(cats,dtype=np.float32)
    to_fill[i] = 1
    return to_fill


def make_sample(session):

    session_meta = np.zeros(3,dtype=np.float32)

    session_len = session[0][2]

    session_meta[0] = np.float32(date_convert(session[0][16]))  # onehot 7
    session_meta[1] = np.float32(session_len)  # int
    session_meta[2] = np.float32(session[0][17])  # bool

    # now make session based features
    # construct first as a list of np.array float32
    rows = []
    for e in session:
        session_position = np.float32(e[1])  # int, maybe normalise to float later
        context_switch = np.float32(e[8])  # bool
        no_pause_before_play = np.float32(e[9])  # bool
        short_pause_before_play = np.float32(e[10])  # bool
        long_pause_before_play = np.float32(e[11])  # bool
        hist_user_behavior_is_shuffle = np.float32(e[14])  # bool

        hist_user_behavior_n_seekfwd = make_onehot((unbounded_int(e[12], 5)),3)  # one hot 3
        hist_user_behavior_n_seekback = make_onehot((unbounded_int(e[13], 5)),3)  # one hot 3
        hour_of_day = make_onehot(e[15],24)  # onehot 24
        time_period_of_day = make_onehot(e[15] // 8,3)  # onehot 3
        context_type = make_onehot(e[18],6)  # onehot 6
        hist_user_behavior_reason_start =  make_onehot(e[19],12)  # onehot 12
        hist_user_behavior_reason_end =  make_onehot(e[20], 11)  # onehot 11

        track = np.float32(e[3]+1) #plus 1 so we dont have a 0 track, which will be used for no track info

        skip_1 = np.float32(e[4])  # bool
        skip_2 = np.float32(e[5])  # bool
        skip_3 = np.float32(e[6])  # bool
        not_skipped = np.float32(e[7])  # bool

        bools = np.array([session_position,context_switch,no_pause_before_play,short_pause_before_play,long_pause_before_play
                        ,hist_user_behavior_is_shuffle],dtype=np.float32)


        oneshots = np.concatenate([hist_user_behavior_n_seekfwd, hist_user_behavior_n_seekback, hour_of_day, time_period_of_day,
                                  context_type, hist_user_behavior_reason_start, hist_user_behavior_reason_end])

        bools_skips = np.array([skip_1,skip_2,skip_3,not_skipped],dtype=np.float32)
        track = np.array([track],dtype=np.float32)

        comb = np.concatenate([bools,oneshots,bools_skips,track])

        rows.append(comb)


    first_part = int(np.floor(session_len / 2))
    second_part = session_len-first_part


    rows = np.stack(rows)


    #fill out input
    input_fill = np.zeros((10,72),dtype=np.float32)
    input_fill[10-first_part:,:] = rows[0:first_part,:72]

    #fill out output, dont contain the bools_skips
    output_fill = np.zeros((10),dtype=np.float32)
    output_fill[0:second_part] = rows[first_part:, 0]


    target = np.zeros((10,2),dtype=np.float32)
    target[0:second_part,0] =  1-rows[first_part:, -4]
    target[0:second_part,1] = rows[first_part:, -4]

    mask = np.zeros(10,dtype=np.float32)
    mask[0:second_part] = 1

    #special input containing all tracks
    tracks_all = np.zeros(20, dtype=np.int32)
    tracks_all[0:session_len] = rows[0:session_len, -1]

    track_input = np.zeros(10, dtype=np.int32)
    track_input[10-first_part:] = rows[0:first_part, -1]

    track_output = np.zeros(10, dtype=np.int32)
    track_output[0:second_part] = rows[first_part:, -1]


    #make additional targets
    additional_target = np.zeros((10,3), dtype=np.float32)
    additional_target[0:second_part, 0] = rows[first_part:, -5]
    additional_target[0:second_part, 1] = rows[first_part:, -3]
    additional_target[0:second_part, 2] = rows[first_part:, -2]


    return input_fill, track_input, output_fill, track_output, mask, target, tracks_all, session_meta, additional_target

def make_iter(path, loop=True):
    c = 0
    while 1:
        files = os.listdir(path)
        shuffle(files)
        for f in files:
            #print(c)
            c += 1
            sessions = pickle.load(open(path+f,"rb"))
            for s in sessions:
                sample = make_sample(s)
                yield sample

        if not loop:
            break
    while 1:
        input_fill = np.zeros((10, 72), dtype=np.float32)
        output_fill = np.zeros((10), dtype=np.float32)
        mask = np.zeros(10, dtype=np.float32)
        tracks_all = np.zeros(20, dtype=np.int32)
        track_input = np.zeros(10, dtype=np.int32)
        track_output = np.zeros(10, dtype=np.int32)
        session_meta = np.zeros(3, dtype=np.float32)
        target = np.zeros((10, 2), dtype=np.float32)
        additional_target = np.zeros((10, 3), dtype=np.float32)

        sample = (input_fill, track_input, output_fill, track_output, mask, target, tracks_all, session_meta, additional_target)
        yield sample

def make_iter_test(files):
    c = 0
    for f in files:
        print(c)
        c += 1
        (sessions_hist, sessions_log) = pickle.load(open(f,"rb"))
        for i in range(len(sessions_hist)):
            session_hist = sessions_hist[i]
            session_log = sessions_log[i]
            #have to make a fake full session
            for i in range(len(session_log)):
                row = np.copy(session_hist[0])
                for e in range(len(row)):
                    row[e] = 0
                row[0:4] = session_log[i][[0,2,3,1]]
                session_hist.append(row)
            (input_fill, track_input, output_fill, track_output, mask, target, tracks_all, session_meta,
             additional_target) = make_sample(session_hist)
            sample = (input_fill, track_input, output_fill, track_output, mask, tracks_all, session_meta)
            yield sample
    while 1:
        input_fill = np.zeros((10, 72), dtype=np.float32)
        output_fill = np.zeros((10), dtype=np.float32)
        mask = np.zeros(10, dtype=np.float32)
        tracks_all = np.zeros(20, dtype=np.int32)
        track_input = np.zeros(10, dtype=np.int32)
        track_output = np.zeros(10, dtype=np.int32)
        session_meta = np.zeros(3, dtype=np.float32)

        sample = (input_fill, track_input, output_fill, track_output, mask, tracks_all, session_meta)
        yield sample


def make_embedding_tracks(path):
    track_data = pickle.load(open(path, "rb"))
    track_np = np.array(track_data, dtype=np.float32)
    track_np = track_np[:,1:]
    track_np = (track_np - np.mean(track_np,axis=0)) / np.std(track_np,axis=0)

    zero_pad = np.zeros((track_np.shape[0]+1,track_np.shape[1]),dtype=np.float32)
    zero_pad[1:,:] = track_np

    return track_np

class model_1:
    def __init__(self, history, history_track, to_predict, to_predict_track, mask, target, tracks, meta, extra_targets,
                 batch_size,
                 embed_size_random, track_combined_size, rnn_songs_num_units, rnn_size, rnn_encoder_layers,
                 dot_size, add_attn_input, rnn_decoder_layers, layer_between_enc_dec, auxilary_loss, layers_track):

        #need to change batch and time axis, currently is batch x time x inp, but need time x batch x inp
        self.history = tf.transpose(history, [1, 0, 2])

        self.to_predict = tf.expand_dims(to_predict,-1)
        self.to_predict = tf.transpose(self.to_predict, [1, 0, 2])
        self.history_track = tf.transpose(history_track, [1, 0])
        self.to_predict_track = tf.transpose(to_predict_track, [1, 0])
        self.all_tracks = tf.transpose(tracks, [1, 0])
        self.meta = meta
        self.extra_targets = extra_targets

        #size vars needed
        self.number_outcomes = target.shape[-1]

        #simple input, with single dimension
        self.mask = mask
        self.target = target

        self.batch_size = batch_size

        #vars in network, maybe pack in a dict if its start getting to much
        self.track_combined_size = track_combined_size
        self.embed_size_random = embed_size_random
        self.rnn_songs_num_units = rnn_songs_num_units
        self.rnn_encoder_layers = rnn_encoder_layers
        self.rnn_decoder_layers = rnn_decoder_layers
        self.dot_size = dot_size
        self.add_attn_input = add_attn_input
        self.layer_between_enc_dec = layer_between_enc_dec
        self.layers_track = layers_track

        self.rnn_size = rnn_size
        self.auxilary_loss = auxilary_loss

    def _make_embedding(self, vocab_size, embedding_size, trainable=True, init=False):
        W = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_size], minval=-0.05, maxval=0.05),
                        trainable=trainable, name="embedding")
        if init:
            embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            embedding_init = W.assign(embedding_placeholder)
            return (W, embedding_placeholder, embedding_init)
        else:
            return W

    def _embedding_lookup(self, states, embedding_1,embedding_2):
        embedded_input_1 = tf.nn.embedding_lookup(embedding_1, states)
        embedded_input_2 = tf.nn.embedding_lookup(embedding_2, states)

        embedded_input = tf.concat([embedded_input_1,embedded_input_2],axis=-1)
        hidden_ = embedded_input
        for t in range(self.layers_track):
            hidden_ = tf.layers.dense(hidden_, self.track_combined_size, activation=tf.nn.relu)

        return hidden_

    def _make_rnn_songs(self, states):
        rnn = tf.contrib.cudnn_rnn.CudnnLSTM(
              num_layers=1,
              num_units=self.rnn_songs_num_units,
              dtype=tf.float32)

        #print(rnn.state_shape(self.batch_size)) rnn(states, initial_state = something)

        outputs, output_states = rnn(states)

        #back to time major
        outputs = tf.transpose(outputs, [1, 0, 2])

        #apply softmax
        track_rep = self._apply_softmax(outputs, "all_tracks_rep")

        return track_rep

    def _apply_softmax(self, output_of_rnn, name):
        logits = tf.layers.dense(output_of_rnn, 1, name=("attention_logit_" + name))
        softed = tf.nn.softmax(logits) * output_of_rnn
        weighted = tf.math.reduce_sum(softed, axis=-2)
        return weighted

    def _make_rnn_history(self, meta, track_rep, hist, hist_track):
        #the meta and track rep is both used to make an initial state for the rnn
        rnn = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=self.rnn_encoder_layers,
            num_units=self.rnn_size,
            dtype=tf.float32)

        make_initial_state_from = tf.concat([meta, track_rep], axis=-1)

        #insert loop and concat on 0 axis to get multiple layers


        i = []
        o = []
        for j in range(self.rnn_encoder_layers):
            internal = tf.layers.dense(make_initial_state_from, self.rnn_size, name="internal_hist" + str(j))
            out = tf.layers.dense(make_initial_state_from, self.rnn_size, name="out_hist" + str(j))
            internal = tf.expand_dims(internal, axis=0)
            out = tf.expand_dims(out,axis=0)
            i.append(internal)
            o.append(out)

        internal = tf.concat(i,axis=0)
        out = tf.concat(o,axis=0)

        initial_state = (internal,out)

        input = tf.concat([hist,hist_track],axis=-1)
        outputs, output_states = rnn(input, initial_state=initial_state)
        return output_states,outputs

    def _make_rnn_predict(self, history_state, to_predict,to_predict_track):
        rnn = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=self.rnn_decoder_layers,
            num_units=self.rnn_size,
            dtype=tf.float32)

        input = tf.concat([to_predict,to_predict_track],axis=-1)
        outputs, output_states = rnn(input, initial_state=history_state)
        return outputs


    def _get_out_LSTM(self, lstm_state):
        (internal, out) = lstm_state
        return out


    def _make_prediction(self, output_rnn):
        hidden_1 = tf.layers.dense(output_rnn, self.rnn_size,activation=tf.nn.relu)
        logits = tf.layers.dense(hidden_1, self.number_outcomes)
        return logits

    def _make_prediction_aux(self, output_rnn):
        hidden_1 = tf.layers.dense(output_rnn, self.rnn_size,activation=tf.nn.relu)
        logits = tf.layers.dense(hidden_1, 3)
        return logits

    def _make_loss(self, logits, target, mask, logits_aux, extra_targets):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=target,
            logits=logits,
            dim=-1,
        )

        primary_loss = tf.reduce_mean(loss * mask)

        if self.auxilary_loss > 0:
            auxilary_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=extra_targets,
                logits=logits_aux,
            )
            auxilary_loss = tf.reduce_sum(auxilary_loss,axis=-1)
            auxilary_loss =  tf.reduce_mean(auxilary_loss * mask)
        else:
            auxilary_loss = 0

        total_loss = primary_loss+auxilary_loss*self.auxilary_loss

        return total_loss


    def make_attention_based_input(self, hist_out, track):
        hist_out = tf.transpose(hist_out,[1,0,2])
        track = tf.transpose(track,[1,0,2])
        tiled_hist_out = tf.reshape(tf.tile(hist_out,[1,1,10]),[self.batch_size,10,10,self.rnn_size])

        hist_out_dot = tf.layers.dense(tiled_hist_out, self.dot_size)
        track_dot = tf.layers.dense(track, self.dot_size)
        track_dot = tf.expand_dims(track_dot, axis=-1)

        logits = tf.matmul(hist_out_dot, track_dot)

        soft = tf.nn.softmax(logits)

        weighted = tiled_hist_out*soft
        hist_out_weigthed = tf.reduce_sum(weighted,axis=-2)
        hist_out_weigthed = tf.transpose(hist_out_weigthed,[1,0,2])

        return hist_out_weigthed


    def transform_encoder_state(self, history_state):
        (internal_stacked, out_stacked) = history_state

        if self.rnn_decoder_layers != self.rnn_encoder_layers:
            print("different size rnns")
            internals = []
            outs = []
            for i in range(self.rnn_encoder_layers):
                internals.append(internal_stacked[i,:,:])
                outs.append(out_stacked[i,:,:])

            internal_concat = tf.concat(internals, axis=-1)
            out_concat = tf.concat(outs,axis=-1)

            internals = []
            outs = []
            for i in range(self.rnn_decoder_layers):
                internals.append(tf.layers.dense(internal_concat,self.rnn_size, name= "between_internal_"+str(i)))
                outs.append(tf.layers.dense(out_concat,self.rnn_size, name= "between_out_"+str(i)))
            internal_return = tf.stack(internals,axis=0)
            out_return = tf.stack(outs,axis=0)

            return (internal_return,out_return)
        elif self.layer_between_enc_dec:
            print("layer between rnns")
            #added rely recently !!!
            internal_stacked = tf.layers.dense(internal_stacked, self.rnn_size, activation=tf.nn.relu)
            out_stacked = tf.layers.dense(out_stacked, self.rnn_size, activation=tf.nn.relu)
            return (internal_stacked,out_stacked)
        else:
            return history_state



    def make_network(self, track_embed, sess):
        #first we make embedding, we make a random embedding, and a fixed embedding for the tracks
        (vocab_size, fixed_size) = track_embed.shape

        track_random_embedding = self._make_embedding(vocab_size,self.embed_size_random, trainable=True, init=False)
        track_fixed_embedding, fixed_place, fixed_init = self._make_embedding(vocab_size,fixed_size, trainable=False, init=True)

        sess.run(fixed_init, feed_dict={fixed_place : track_embed})

        history_track_embed = self._embedding_lookup(self.history_track,track_random_embedding,track_fixed_embedding)
        to_predict_embed = self._embedding_lookup(self.to_predict_track,track_random_embedding,track_fixed_embedding)
        all_track_embed = self._embedding_lookup(self.all_tracks, track_random_embedding,track_fixed_embedding)

        track_rep = self._make_rnn_songs(all_track_embed)

        history_state, hist_all_outputs = self._make_rnn_history(self.meta, track_rep, self.history, history_track_embed)

        #For all the best performing runs, this is simple an identity function, could not get this idea to work
        history_state = self.transform_encoder_state(history_state)


        # attn input, this does not improve performance, but lead to overfitting, is not used for the best performing runs
        if self.add_attn_input:
            attn_input = self.make_attention_based_input(hist_all_outputs, to_predict_embed)
            input_to_pred = tf.concat([to_predict_embed, attn_input], axis=-1)
            rnn_pred_out_state = self._make_rnn_predict(history_state, self.to_predict,input_to_pred)
        else:
            rnn_pred_out_state = self._make_rnn_predict(history_state, self.to_predict, to_predict_embed)



        #return to batch x time x feat
        rnn_pred_out_state = tf.transpose(rnn_pred_out_state, [1, 0, 2])

        logits = self._make_prediction(rnn_pred_out_state)
        # Currently do the prediction using a softmax with 2 outcomes (skit not skip). Is legacy from earlier where we worked with multiple outcones,
        # The final solution should just have been a single sigmoid as written in paper, but did no refactor.
        pred = tf.nn.softmax(logits)

        #acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred, axis=-1),  tf.argmax(target, axis=-1)), dtype=tf.float32) * self.mask) / tf.reduce_sum(self.mask)

        if self.auxilary_loss > 0: # for all best performing runs, auxilary loss did not improve performance
            logits_aux = self._make_prediction_aux(rnn_pred_out_state)
        else:
            logits_aux = None

        loss = self._make_loss(logits, self.target, self.mask, logits_aux, self.extra_targets)

        return pred, loss



def mean_average_accuracy(pred, target, mask):
    pred = np.argmax(pred, axis=-1)
    target = np.argmax(target, axis=-1)
    elements = np.sum(mask, axis=-1)
    mean_average_accuracy = np.zeros((pred.shape[0],pred.shape[1]))
    correct = np.equal(pred, target)
    for i in range(pred.shape[1]):
        mean_average_accuracy[:,i] = np.sum(correct[:,0:i+1],axis=-1) / (i+1)

    aa = np.mean(np.sum((mean_average_accuracy*correct*mask),axis=-1) / elements)
    return aa


def run_on_test(path_to_test_files,path_track, model_path,output_path,vars_dict):
    print("running on test", model_path)
    files = (glob.glob(path_to_test_files + "*.p"))
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    print(files)

    #just for debugging, set to False for normal use
    use_train_files = False
    train_files = path_to_test_files

    tf.reset_default_graph()
    sess = tf.Session()

    if not use_train_files:
        output_t = (tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.int32, tf.float32)
        output_s = (tf.TensorShape([10, 72]), tf.TensorShape([10]), tf.TensorShape([10]), tf.TensorShape([10]),
                    tf.TensorShape([10]), tf.TensorShape([20]), tf.TensorShape([3]))

        dataset = tf.data.Dataset.from_generator(make_iter_test, output_types=output_t, output_shapes=output_s,args=[files])
    else:
        output_t = (
        tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32)
        output_s = (tf.TensorShape([10, 72]), tf.TensorShape([10]), tf.TensorShape([10]), tf.TensorShape([10]),
                    tf.TensorShape([10]), tf.TensorShape([10, 2]), tf.TensorShape([20]), tf.TensorShape([3]),
                    tf.TensorShape([10, 3]))
        dataset = tf.data.Dataset.from_generator(make_iter, output_types=output_t, output_shapes=output_s,
                                                 args=[train_files, False])

    dataset = dataset.batch(vars_dict["batch_size"])
    dataset = dataset.prefetch(6000)
    #dataset = dataset.shuffle(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()

    if not use_train_files:
        target = tf.placeholder(shape=(vars_dict["batch_size"],10,2),dtype=tf.float32) #not used
        target_aux = None#tf.placeholder(shape=(vars_dict["batch_size"],10,3),dtype=tf.float32) #not used
        (history, history_tracks, to_predict_on, to_predict_on_track, mask, tracks, meta) = iterator.get_next()
    else:
        target_1 = tf.placeholder(shape=(vars_dict["batch_size"],10,2),dtype=tf.float32) #not used
        target_aux = None
        (history, history_tracks, to_predict_on, to_predict_on_track, mask, target, tracks, meta,
         _) = iterator.get_next()

    model = model_1(history, history_tracks, to_predict_on, to_predict_on_track, mask, target, tracks, meta,target_aux, **vars_dict)

    track = make_embedding_tracks(path_track)
    pred, loss = model.make_network(track, sess)

    saver = tf.train.Saver()

    #load old model
    saver.restore(sess, model_path)

    def make_answer(preds,mask):
        preds = np.argmax(preds, axis=-1)
        length = preds.shape[0]
        to_output = []
        for i in range(length):
            row_mask = mask[i,:]
            row_preds = preds[i,:]
            if np.sum(row_mask) > 0:
                r = row_preds[0:int(np.sum(row_mask))]
                to_output.append(list(r))

        return to_output


    output = []
    while 1:
        if not use_train_files:
            pred_current, mask_current = sess.run([pred, mask])
        else:
            pred_current, mask_current, target_current = sess.run([pred, mask, target])

            #c print(mean_average_accuracy(pred_current, target_current, mask_current))
        to_append = make_answer(pred_current,mask_current)
        output += to_append
        if np.sum(mask_current[:]) == 0:
            break

    def save_submission(output,output_path):
        with open(output_path,"w") as f:
            for l in output:
                line = ''.join(map(str,l))
                f.write(line + '\n')
        print('submission saved to {}'.format(output_path))

    save_submission(output,output_path + "submission")

def train_network(path_to_train,path_to_val,path_track,path_to_res,outer_dict,vars_dict, hours_between_models=3.0):
    print("training")
    print(path_to_res)

    tf.reset_default_graph()
    sess = tf.Session()

    #setup data
    output_t = (tf.float32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.float32,tf.float32)
    output_s = (tf.TensorShape([10,72]), tf.TensorShape([10]),tf.TensorShape([10]),tf.TensorShape([10]),
                tf.TensorShape([10]), tf.TensorShape([10,2]), tf.TensorShape([20]), tf.TensorShape([3]),
                tf.TensorShape([10, 3]))
    dataset = tf.data.Dataset.from_generator(make_iter, output_types=output_t, output_shapes=output_s, args=[path_to_train])
    dataset = dataset.batch(vars_dict["batch_size"])
    dataset = dataset.prefetch(500)
    training_iterator = dataset.make_one_shot_iterator()
    training_handle = sess.run(training_iterator.string_handle())

    dataset_val = tf.data.Dataset.from_generator(make_iter, output_types=output_t, output_shapes=output_s, args=[path_to_val])
    dataset_val = dataset_val.batch(vars_dict["batch_size"])
    dataset_val = dataset_val.prefetch(6000)
    validation_iterator = dataset_val.make_one_shot_iterator()
    validation_handle = sess.run(validation_iterator.string_handle())

    handle = tf.placeholder(tf.string, shape=[], name="handle_for_iterator")
    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes) #dataset.make_one_shot_iterator() #from_string_handle use this for switching betwwen, and then make some
                                                # way we can handle limited input

    (history, history_tracks, to_predict_on, to_predict_on_track, mask, target, tracks, meta, extra_targets) = iterator.get_next()

    #setup model
    model = model_1(history, history_tracks, to_predict_on, to_predict_on_track, mask, target, tracks, meta, extra_targets, **vars_dict)
    #tenp track until make function
    track = make_embedding_tracks(path_track)
    pred, loss = model.make_network(track, sess)
    optimizer = tf.train.AdamOptimizer(learning_rate=outer_dict["lr"])
    train_step = optimizer.minimize(loss)



    #logging data
    summary_period = 100
    val_period = 30000

    pythonVar = tf.placeholder(tf.float32, [])
    tensorboardVar = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
    update_tensorboardVar = tensorboardVar.assign(pythonVar)
    sum1 = tf.summary.scalar("mean-average-accuracy", tensorboardVar)
    sum2 = tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge([sum1, sum2])

    pythonVar_val = tf.placeholder(tf.float32, [])
    tensorboardVar_val = tf.Variable(0, dtype=tf.float32, name="tensorbardVar")
    update_tensorboardVar_val = tensorboardVar_val.assign(pythonVar_val)
    sum3 = tf.summary.scalar("mean-average-accuracy_val", tensorboardVar_val)
    summary_op_val = tf.summary.merge([sum3])



    summary_writer = tf.summary.FileWriter(path_to_res + "overview/", sess.graph)

    config_summary = tf.summary.text('TrainConfig_runner', tf.convert_to_tensor(as_matrix(vars_dict)), collections=[])
    outer_summary = tf.summary.text('Outer_pars', tf.convert_to_tensor(as_matrix(outer_dict)), collections=[])
    summary_config_op = tf.summary.merge([config_summary,outer_summary])
    summary_writer.add_summary(sess.run(summary_config_op))


    init = tf.global_variables_initializer()
    sess.run(init)


    saver = tf.train.Saver()
    #model_path = "C:\\Users\\christian\\Dropbox\\phd\\Projects\\spotify\\Res\\test\\first_run-129"
    #saver.restore(sess, model_path)

    #loops vars
    aas = []
    counter = 0
    counter_val = 0
    current_time = time.time()
    t = 3
    log_i = 0
    time_s = time.time()
    while 1:
        (pred_current, loss_current, target_current, mask_current, _) = sess.run([pred, loss, target, mask, train_step],
                                                                                 feed_dict={handle: training_handle})
        aa = mean_average_accuracy(pred_current, target_current, mask_current)
        aas.append(aa)
        counter += 1
        counter_val +=1
        if counter > summary_period:
            _ = sess.run([update_tensorboardVar], feed_dict={pythonVar: np.mean(aas),handle: training_handle})
            result = sess.run(summary_op, feed_dict={handle: training_handle})
            summary_writer.add_summary(result, log_i)
            log_i += 1
            aas = []
            counter = 0
            if (time.time()-current_time)/60/60 > hours_between_models:
                saver.save(sess, path_to_res + "first_run", global_step=t)
                t += 3
                current_time = time.time()
        if counter_val > val_period:
            val_ress = []
            for i in range(1000):
                (pred_current, loss_current, target_current, mask_current) = sess.run(
                    [pred, loss, target, mask],
                    feed_dict={handle: validation_handle})
                aa = mean_average_accuracy(pred_current, target_current, mask_current)
                val_ress.append(aa)

            _ = sess.run([update_tensorboardVar_val], feed_dict={pythonVar_val: np.mean(val_ress), handle: validation_handle})
            result = sess.run(summary_op_val, feed_dict={handle: validation_handle})
            summary_writer.add_summary(result, log_i)
            print("val:", np.mean(val_ress))
            counter_val = 0

if __name__ == "__main__":
    mode = "train"

    #network parameters, outer dict is for the learning rate
    outer_dict = {
        "lr": 0.0005,
    }
    #vars dict is for all parameters used inside the network
    vars_dict = {
        "batch_size": 300,
        "embed_size_random": 50,
        "track_combined_size": 350,
        "rnn_songs_num_units": 100,
        "rnn_size": 500,
        "rnn_encoder_layers": 2,
        "rnn_decoder_layers": 2,
        "dot_size": 150,
        "add_attn_input": False,
        "layer_between_enc_dec": False,
        "auxilary_loss": -1,
        "layers_track": 1,
    }

    #train
    path_track = "data/track_features/track_data.pickle"
    if mode == "train":
        path_to_train = "data/training_set_proc/"
        path_to_val = "training_set_proc_val" #TAKE 1 or more files at random from path_to_train to use as validation, only need to be done once and used across all training
        path_to_res = "data/Models/1/"
        train_network(path_to_train, path_to_val, path_track, path_to_res, outer_dict, vars_dict)
    #test
    else:
        path_to_test_files = "data/test_set_proc/"
        model_path = "data/Models/1/first_run-3" #point to the model name (name before the .data, .index .meta of the saved model)
        output_path = "data/Results/1/"
        run_on_test(path_to_test_files,path_track, model_path, output_path,vars_dict)