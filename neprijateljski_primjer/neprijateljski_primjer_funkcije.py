import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def get_label(i, cifar):
    CIFAR_MAP = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if cifar:
        return CIFAR_MAP[i]
    else:
        return f"Predviđen broj: {str(i)}"

def plot(wrong_predictions, n = 5, cifar = True):
    fig, axs = plt.subplots(max(2, n), 2, figsize=(10, n * 10))

    for i in range(n):
        axs[i, 0].imshow(torch.clamp(wrong_predictions[i][0][0], min = 0.0, max = 1.0).detach().cpu().permute(1, 2, 0))
        axs[i, 0].set_title(f"{get_label(wrong_predictions[i][1].item(), cifar)} - {wrong_predictions[i][4]}")
        axs[i, 1].imshow(torch.clamp(wrong_predictions[i][2][0], min = 0.0, max = 1.0).detach().cpu().permute(1, 2, 0))
        axs[i, 1].set_title(f"{get_label(wrong_predictions[i][3].item(), cifar)} - {wrong_predictions[i][5]}")

def plot_single(img, gray = False):
    plt.figure()
    if gray: plt.imshow(img, cmap='gray')
    else: plt.imshow(img)

def get_guess_and_prob(output):
    guess = output.max(1, keepdim=True)[1][0]
    guess_prob = round(nn.functional.softmax(output[0], dim=0)[guess].item(), 4)
    return guess, guess_prob

def attack(model, dataset, eps = 0.025, n = 100):
    model.eval()

    correct = 0
    correct_pert = 0
    all = 0
    loss_f = nn.CrossEntropyLoss()

    wrong_predictions = []

    for (image, label) in dataset:
        # Učitavamo slike i labele u GPU memoriju
        image = image.to('cuda:0')
        label = label.to('cuda:0')

        # Želimo gradijente i za same slike (efektivno ih tretiramo kao parametre)
        image.requires_grad = True

        output = model(image)
        loss = loss_f(output, label)
        loss.backward()

        old_guess, old_guess_prob = get_guess_and_prob(output)
        correct += old_guess == label
        all += 1

        # Perturbiramo sliku u smjeru najvećeg porasta gubitka
        new_image = image + eps*image.grad.data.sign()

        pert_out = model(new_image)
        guess, guess_prob = get_guess_and_prob(pert_out)
        correct_pert += guess == label

        if guess != label:
          wrong_predictions.append([image, label, new_image, guess, old_guess_prob, guess_prob])

        if correct > n:
            print(correct/all)
            print(correct_pert/all)
            break

    return wrong_predictions

def class_attack(model, dataset, target_class, eps = 0.025, n = 1000):
    # CIFAR10 class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    #                         0           1       2     3     4    5    6     7       8     9
    target_class = torch.as_tensor([target_class]).to('cuda:0')
    model.eval()

    correct = 0
    correct_pert = 0
    all = 0
    loss_f = nn.CrossEntropyLoss()

    wrong_predictions = []

    for (image, label) in dataset:
        # Učitavamo slike i labele u GPU memoriju
        image = image.to('cuda:0')
        label = label.to('cuda:0')
        # Želimo gradijente i za same slike (efektivno ih tretiramo kao parametre)
        image.requires_grad = True
        output = model(image)
        # Gubitak računamo na temelju ciljane klase
        loss = loss_f(output, target_class)
        loss.backward()

        old_guess, old_guess_prob = get_guess_and_prob(output)
        correct += old_guess == label
        all += 1

        # Perturbiramo sliku
        new_image = image - eps*image.grad.data.sign()

        pert_out = model(new_image)
        guess, guess_prob = get_guess_and_prob(pert_out)
        correct_pert += guess == label

        # Gledamo slike koje je model prvotno točno labelirao, ali sada krivo
        if guess != label and guess == target_class and old_guess == label:
            wrong_predictions.append([image, label, new_image, guess, old_guess_prob, guess_prob])

        if correct > n:
            print(correct/all)
            print(correct_pert/all)
            break

    return wrong_predictions

def selected_grad_attack(model, dataset, eps = 0.5, n = 100, p = 0.1):
    model.eval()

    correct = 0
    correct_pert = 0
    all = 0
    loss_f = nn.CrossEntropyLoss()

    wrong_predictions = []

    for (image, label) in dataset:
        # Učitavamo slike i labele u GPU memoriju
        image = image.to('cuda:0')
        label = label.to('cuda:0')

        # Želimo gradijente i za same slike (efektivno ih tretiramo kao parametre)
        image.requires_grad = True

        output = model(image)
        loss = loss_f(output, label)
        loss.backward()

        old_guess, old_guess_prob = get_guess_and_prob(output)
        correct += old_guess == label
        all += 1

        grads = image.grad.reshape(-1,) # 1x3xAxB u 1*3*A*B
        abs_grads = torch.abs(grads)
        k = int(abs_grads.numel() * p)
        kth_biggest_grad = abs_grads.kthvalue(abs_grads.numel() - k).values.item()
        selected_grads = abs_grads.gt(kth_biggest_grad).int().reshape(image.shape)

        # Perturbiramo sliku
        new_image = image + eps * image.grad.sign() * selected_grads

        pert_out = model(new_image)
        guess, guess_prob = get_guess_and_prob(pert_out)
        correct_pert += guess == label

        if guess != label:
          wrong_predictions.append([image, label, new_image, guess, old_guess_prob, guess_prob])

        if correct > n:
            print(correct/all)
            print(correct_pert/all)
            break

    return wrong_predictions

def generate_images(model, shape, label, eps = 0.025, n = 100):
    class S_loss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(S_loss, self).__init__()

        def forward(self, input, target):
            return -input[0][target]

    model.eval()
    label = torch.as_tensor([label]).to('cuda:0')
    loss_f = S_loss()

    wrong_predictions = []

    # Stvaramo sliku šuma
    new_image = torch.zeros(shape) + torch.randn(shape)*0.1
    new_image = new_image.to('cuda:0')
    new_image.requires_grad = True

    optimizer = optim.SGD([new_image], lr=eps, momentum=0.9, weight_decay=1e-4)

    for _ in range(n):
        optimizer.zero_grad()

        output = model(new_image)
        loss = loss_f(output, label)
        loss.backward()

        optimizer.step()
    print(loss)
    return new_image

def show_grads(wrong_predictions, n = 5, p = 0.05):
    fig, axs = plt.subplots(max(2, n), 2, figsize=(10, n * 10))
    for i in range(n):
        image = wrong_predictions[i][0]
        k = int(p * image.nelement())
        grads = image.grad.reshape(-1,)
        abs_grads = torch.abs(grads)
        kth_biggest_grad = abs_grads.kthvalue(abs_grads.numel() - k).values.item()
        selected_grads = abs_grads.gt(kth_biggest_grad).int().reshape(image.shape) * 1.0
        axs[i, 0].imshow(torch.clamp(image[0], min = 0.0, max = 1.0).detach().cpu().permute(1, 2, 0))
        axs[i, 0].set_title(f"Originalna slika")
        axs[i, 1].imshow(torch.clamp(selected_grads[0], min = 0.0, max = 1.0).detach().cpu().permute(1, 2, 0))
        axs[i, 1].set_title(f"{int(p * 100)}% najznacajnijih gradijenata")
