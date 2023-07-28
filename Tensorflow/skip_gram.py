import io 
import re 
import string 
import tqdm 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers 

SEED = 42
AUTOTUNE =tf.data.AUTOTUNE


sentence = "The wide road shimmered in the hot sun"
tokens =  list(sentence.lower().split())
print(len(tokens))


vocab,index={},1
vocab['<pad>'] = 0
for token in tokens:
  if token not in vocab:
    vocab[token]=index 
    index+=1
vocab_size=len(vocab)
print(vocab ) 


inverse_vocab={idx: token for token,idx in vocab.items()}
print(inverse_vocab)


example_sequence = [vocab[word] for word in tokens]
print(example_sequence)


window_size=2 
positive_skip_grams,_=tf.keras.preprocessing.sequence.skipgrams(example_sequence,vocabulary_size=vocab_size,window_size=window_size,negative_samples=0)


target_word,context_word = positive_skip_grams[0]

num_ns=4 

context_class =tf.reshape(tf.constant(context_word,dtype="int64"),(1,1))


negative_sampling_candidates,_,_=tf.random.log_uniform_candidate_sampler(true_classes=context_class,num_true=1,num_sampled=num_ns,unique=True,range_max=vocab_size,seed=SEED,name="negative_Sampling")

squeezed_context_class = tf.squeeze(context_class,1)

context=tf.concat([squeezed_context_class,negative_sampling_candidates],0)


labels=tf.constant([1]+[0]*num_ns,dtype="int64")

target=target_word




def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)
                         
    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels
          

path_to_file = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text_ds=tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x),bool))

vocab_size=4096
sequence_length = 10

vectorize_layer=layers.TextVectorization(standardize=coustom_standarization,max_tokens=vocab_size,output_mode='int',output_sequence_length=sequence_length)


def coustom_standarization(input_data):
   lowercase = tf.strings.lower(input_data)
   return tf.strings.regex_replace(lowercase,'[%s]'% re.escape(string.punctuation),'') 


vectorize_layer.adapt(text_ds.batch(1024))

inverse_vocab=vectorize_layer.get_vocabulary()



text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()


sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))



targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED)

targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")



BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)


dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)



#model 

class Word2Vec(tf.keras.Model):

  def __init__(self,vocab_size,embedding_dim):
    super(Word2Vec,self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,embedding_dim,
                                             input_length=1,
                                             name="w2v_embedding")
    self.context_embedding=layers.Embedding(vocab_size,embedding_dim,input_length=num_ns+1)
  



  def call(self,pair):
    target,context=pair

    if len(target.shape)==2:
      target = tf.squeeze(target,axis=1)
    
    word_emb=self.target_embedding(target)

    context_emb=self.context_emb(context)

    dots = tf.einsum('be,bce->bc',word_emb,context_emb)
    return dots 



def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
  
  
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])



Word2Vec.fit(dataset, epochs=20)