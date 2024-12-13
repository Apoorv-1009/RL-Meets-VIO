from typing import Any, Callable, Tuple, Type, Union, Dict, List

from gymnasium import spaces
import torch
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


class PerceiverI(nn.Module):
    def __init__(self, kv_dim, num_queries, num_heads=4, dim_head=32):
        super().__init__()
        self.num_heads = num_heads
        query_dim = dim_head * self.num_heads

        self.latents = nn.Parameter(torch.randn(num_queries, query_dim))

        self.kv_layers = nn.Sequential(nn.Linear(kv_dim, query_dim),
                                       nn.LayerNorm(query_dim))
        self.multihead_attention = nn.MultiheadAttention(query_dim, self.num_heads, batch_first=True,
                                                         kdim=query_dim, vdim=query_dim)

    def forward(self, inputs, key_padding_mask):
        b = inputs.shape[0]
        inputs = self.kv_layers(inputs.flatten(0, 1)).unflatten(0, [b, -1])
        batch_latent = torch.tile(self.latents[None, :], (b, 1, 1))
        x_attn = self.multihead_attention(query=batch_latent,
                                          key=inputs,
                                          value=inputs,
                                          key_padding_mask=key_padding_mask)[0]

        return x_attn


class AttentionNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        encoder_kwargs: Dict,
    ):
        super().__init__()

        self.num_queries = 4
        self.num_heads = 4
        self.dim_head = 16
        self.variable_flattened_dim = self.num_queries * self.num_heads * self.dim_head

        self.obs_dim_variable = encoder_kwargs['obs_dim_variable']
        self.obs_dim_fixed = encoder_kwargs['obs_dim_fixed']
        self.variable_feature_dim = encoder_kwargs['variable_feature_dim']
        self.critique_dim = encoder_kwargs['critique_dim']

        # Keypoint encoder
        self.variable_encoder = PerceiverI(encoder_kwargs['variable_feature_dim'], self.num_queries,
                                           num_heads=self.num_heads, dim_head=self.dim_head)

        # Standard Stable Baselines Network
        last_layer_dim_pi = self.variable_flattened_dim + encoder_kwargs['obs_dim_fixed']

        last_layer_dim_vf = last_layer_dim_pi + self.critique_dim
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

    def forward_variable(self, features: torch.Tensor) -> torch.Tensor:
        b = features.shape[0]
        variable_features = features[:, self.obs_dim_fixed:].reshape([b, -1, self.variable_feature_dim])
        valid_mask = torch.abs(variable_features.sum([1, 2])) != 0
        out_attn_features = torch.zeros([b, self.variable_flattened_dim]).to(device=features.device)
        if valid_mask.sum() != 0:
            key_padding_mask = torch.abs(variable_features[valid_mask, :, :].sum(-1)) == 0
            out_attn_features[valid_mask] = self.variable_encoder.forward(variable_features[valid_mask, :, :],
                                                                          key_padding_mask).flatten(1,2)

        return torch.cat([features[:, :self.obs_dim_fixed], out_attn_features], dim=1)
        
        # batch_size = out_attn_features.shape[0]
        # prev_output_dim = out_attn_features.shape[1]
        # # print("batch_size: ",batch_size)
        # # print("prev_output_dim: ",prev_output_dim)
        # # print("obs_dim_fixed: ",self.obs_dim_fixed)
        # cross_attention_pretext = nn.Linear(out_attn_features.shape[1],self.obs_dim_fixed).to(device=features.device)
        # cross_attention_block = nn.MultiheadAttention(self.num_heads,self.num_heads,batch_first=True,kdim=self.num_heads,vdim=self.num_heads).to(device=features.device)
        # cross_attention_posttext = nn.Linear(self.num_heads*self.obs_dim_fixed,prev_output_dim+self.obs_dim_fixed).to(device=features.device)
        # cross_attention_gelu = nn.GELU().to(device=features.device)
        
        # out_attn_features_ca = cross_attention_pretext(out_attn_features).reshape(-1,self.obs_dim_fixed,1)
        # ca_output = cross_attention_block(query=torch.tile(features[:,:self.obs_dim_fixed].reshape(-1,self.obs_dim_fixed,1),[1,1,self.num_heads]).to(device=features.device),
        #                                   key=torch.tile(out_attn_features_ca.to(device=features.device),[1,1,self.num_heads]),
        #                                   value=torch.tile(out_attn_features_ca.to(device=features.device),[1,1,self.num_heads]))[0]
        # # print("ca_output_shape: ",ca_output.shape)
        # ca_output_post_process = cross_attention_gelu(cross_attention_posttext(ca_output.reshape(batch_size,-1)))

        # return ca_output_post_process

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        agent_features = features[:, :-self.critique_dim]
        agent_features = self.forward_variable(agent_features)

        critique_features = torch.cat([agent_features, features[:, -self.critique_dim:]], dim=1)
        return self.policy_net(agent_features), self.value_net(critique_features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        agent_features = features[:, :-self.critique_dim]
        agent_features = self.forward_variable(agent_features)
        return self.policy_net(agent_features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        agent_features = features[:, :-self.critique_dim]
        agent_features = self.forward_variable(agent_features)

        critique_features = torch.cat([agent_features, features[:, -self.critique_dim:]], dim=1)
        return self.value_net(critique_features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        self.encoder_kwargs = kwargs['encoder_kwargs']
        del kwargs['encoder_kwargs']

        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AttentionNetwork(self.features_dim,  self.net_arch, self.activation_fn, self.encoder_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                encoder_kwargs=self.encoder_kwargs,
            )
        )
        return data