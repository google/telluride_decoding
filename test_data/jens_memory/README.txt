This test file was created by shortening the number of trails for one subject.
This was done using the following matlab code to create a smaller test file.


>> load data_01.mat
>> data

data =

  struct with fields:

        trial: {1×40 cell}
    trialinfo: [40×1 double]
         time: {1×40 cell}
        label: {70×1 cell}
      fsample: 128

>> short = data.trial(1:5);
>> data.trial = short;
>> data

data =

  struct with fields:

        trial: {1×5 cell}
    trialinfo: [40×1 double]
         time: {1×40 cell}
        label: {70×1 cell}
      fsample: 128

>> save data_01short.mat data
