import pickle

a = [1]
print('a: ',a)

saved = pickle.dumps(a)
print('saved a:',saved)
b = pickle.loads(saved)
print('b: ',b)
b.append(2)
savedLoadArray = pickle.dumps(b)
print('saved b: ',savedLoadArray)
c = pickle.loads(savedLoadArray)
print('c: ',c)
