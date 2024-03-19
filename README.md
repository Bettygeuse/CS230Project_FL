# CS230Project_FL

### AWS EC2 Setup
(Instruction writing refered to https://github.com/Ryanhilde/cs230-hdfs/blob/main/launch.md)
#### 1. Create a key pair
 - Go to https://aws.amazon.com/ -> `Sign In to the Console`.
 - `Sevices` -> `Compute` -> `EC2`.
 - Left menu -> `Network & Security` -> `Key Pairs`.
 - `Create key pair`
   - Input `Name`, e.g., `cs230`.
   - Select `Private key file format`, use `.pem` for Linux/Mac or `.ppk` for Windows.
   - Confirm `Create key pair`.
   - Save the file to your local computer for future use.

#### 2. Create Amazon EC2 Ubuntu instances
 - Left menu -> `Instances` -> `Instances`.
 - `Launch instances`
   - Under tab `Application and OS Images (Amazon Machine Image)`, select `Ubuntu`. (By default, the AMI is `Ubuntu Server 22.04 LTS (HVM), SSD Volume Type`)
   - Under tab `Instance type`, select `t2.micro` (by default).
   - Under tab `Key pair`, select the one created in Step 1.
   - Under tab `Configure storage`, type `30` GIB and choose `gp3`.
   - On the right `Summary`, select `6` in `Number of instances`, then `Launch instance`.
   - Click `View all instances` and wait for the 6 instances' statuses become `running`.
 - Upload the repository to the instance.
 ```bash
scp -i ../cs230.pem -r CS230PROJECT_FL/ ubuntu@<public_ip>
 ```
 - Use terminal to connect to the instances.
   - On Linux/Mac, 
```bash
ssh -i <pem_filepath> ubuntu@<public_ip>

# `<pem_filepath>` is the path to the `.pem` file downloaded in Step 1, 
# `<public_ip>` is the `Public IPv4 address` in the details of each instance.
# NOTE: If see `Permission denied (publickey).` error, 
#       use `chmod 400 <pem_filepath>` to update the permissions to the `.pem` file.
```

#### 3. Dependancy Installation
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


#### 4. Setting up password-less SSH login between instances
 - Assign names to the 3 instances: `Master`, `Worker1`, `Worker2`.
 - Generate the public key and private key on all 3 instances.
```bash
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
```
 - Show the public key on all 3 instances.
```bash
cat .ssh/id_rsa.pub
```
 - Copy and append all 3 public keys to all 3 instances' `.ssh/authorized_keys` file.
 - Verify that you can ssh login in all the following directions without password.

On `Master` run the following,
```bash
# From Master to Master
ssh <public-ip-of-master>
ssh <private-ip-of-master>

# From Master to Worker1
ssh <public-ip-of-worker1>
ssh <private-ip-of-worker1>

# From Master to Worker2
ssh <public-ip-of-worker2>
ssh <private-ip-of-worker2>
```
On `Worker1` run the following,
```bash
# From Worker1 to Worker1
ssh <public-ip-of-worker1>
ssh <private-ip-of-worker1>

# From Worker1 to Master
ssh <public-ip-of-master>
ssh <private-ip-of-master>
```
On `Worker2` run the following,
```bash
# From Worker2 to Worker2
ssh <public-ip-of-worker2>
ssh <private-ip-of-worker2>

# From Worker2 to Master
ssh <public-ip-of-master>
ssh <private-ip-of-master>
```

#### 5. Framework Configuration and Execution
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

### Data Preparation
- Create a folder called './data' in the root directory.
- Download and unzip the data in './data'
- Change the condition on line 13 to process the data with different conditions.
- Run the command
  ```
  python3 data_prepare.py
  ```
