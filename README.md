# CS230Project_FL

#### Dependency Installation
(On all instances)
 - Install flower, pytorch libraries using the following commands.
```bash
sudo apt update
sudo apt install python3-pip
pip3 install torch torchvision --no-cache-dir ## Note that the memory on instances are too small, we need to add --no-cache-dir flag
pip3 install flwr==1.4.0
pip3 install hydra-core
pip3 install ray==1.11.1 # Required only when running simulation
```

### Data Preparation
- Run the command below to finish dataset setup
  ```
  cd datasets/SumoSimulation
  python3 data_preprocess.py
  ```

#### Framework Configuration and Execution
 - On all client nodes: Change the ip in client.py to the private IP of the server node.
 - Flower uses port 8080 to communicate between nodes. Allow access through port 8080 from client nodes.
    - HINT: Refer here for instructions on how to add an IP property to a configuration: https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/conf/Configuration.html 
    - **Note:** For the default network security setting of AWS instance, port `8080` on `Master` node is not accessible from worker nodes. 
    - To allow access, go to the `AWS Console` -> `EC2 instances` -> Click `Master` instance -> `Security` -> Click the security group link -> `Edit inbound rules` -> `Add rule` -> `Type = Custom TCP`, `Port range = 8080`, `Source = 172.31.0.0/16` -> `Save rules`.


 - change the config file `conf/base.yaml` of the server and client parameters. For testing on a real machine, please change the below parameters:
    - client_id: must start from 0 and should be sequantially assigned different number to every client
    - server_address: IP, port of the server identified by a client
    - num_clients: the total number of the client (for training data partitions)
    - uniform_data_distribution: True for uniform data distribution, False for heterogenous data distribution

 - On the server, run the server script using the below command:
 ```bash
python3 server.py
```
 - Wait until the initialization of the server script finishes (Should show "Requesting initial parameters from one random client"). On each client, run the server script using the below command:
 ```bash
python3 client.py
```
- Note that the server only starts when there are equal to or more than 2 client nodes. Wait until the training process finishes.
